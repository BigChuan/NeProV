import os
import json
import test
import tqdm
import torch
import numpy as np
from flags import parser
from utils import load_args, set_seed
from dataset import CompositionDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from model.configure_model import configure_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate(model, dataset, args, verbose=False):
    model.eval()
    evaluator = test.Evaluator(dataset, verbose=verbose)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt = test.predict_logits(model, dataset, args)
    test_stats = test.test(dataset, evaluator, all_logits, all_attr_gt, all_obj_gt, all_pair_gt)
    test_saved_results = {}
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    return test_saved_results, result


def _append_eval_log(log_path, message):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def _move_batch_to_device(batch):
    moved = []
    for item in batch:
        if isinstance(item, tuple):
            moved.append(item)
        else:
            moved.append(item.to(device))
    return moved


def _build_postfix(loss_items, epoch_train_losses):
    postfix = {"loss": f"{np.mean(epoch_train_losses[-50:]):.4f}"}
    if loss_items is None:
        return postfix

    postfix.update({
        "att": f"{loss_items['att']:.4f}",
        "obj": f"{loss_items['obj']:.4f}",
        "com": f"{loss_items['com']:.4f}",
        "glb": f"{loss_items['glb']:.4f}",
        "reg": f"{loss_items.get('reg', 0.0):.4f}",
        "gt": f"{loss_items.get('gt', 0.0):.3f}",
        "so": f"{loss_items.get('sobj', 0.0):.3f}",
        "sa": f"{loss_items.get('sattr', 0.0):.3f}",
        "ot": f"{loss_items.get('other', 0.0):.3f}",
        "attg": f"{loss_items.get('att_g', 0.0):.3f}",
        "objg": f"{loss_items.get('obj_g', 0.0):.3f}",
    })
    return postfix


def train_model(model, scheduler, optimizer, args, train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    best_metric = 0
    best_loss = 1e5
    best_epoch = 0

    train_losses = []
    val_results = []
    history = []
    eval_log_path = os.path.join(args.save_path, "eval.log")
    if os.path.exists(eval_log_path):
        os.remove(eval_log_path)
    _append_eval_log(eval_log_path, f"args: {args}")

    scaler = torch.amp.GradScaler(device, enabled=True)
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm.tqdm(total=len(train_loader), desc="epoch % 3d" % (epoch + 1))
        epoch_train_losses = []
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = _move_batch_to_device(batch)
            with torch.amp.autocast(device, enabled=True):
                loss = model(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_train_losses.append(loss.item())
            loss_items = getattr(model, "latest_loss_items", None)
            if loss_items is None and hasattr(model, "module"):
                loss_items = getattr(model.module, "latest_loss_items", None)

            progress_bar.set_postfix(_build_postfix(loss_items, epoch_train_losses))
            progress_bar.update()

            if loss_items is not None:
                progress_bar.write(
                    f"epoch {epoch+1} step {idx+1}/{len(train_loader)} | "
                    f"total {loss_items['total']:.4f} | att {loss_items['att']:.4f} | "
                    f"obj {loss_items['obj']:.4f} | com {loss_items['com']:.4f} | glb {loss_items['glb']:.4f} | "
                    f"reg {loss_items.get('reg', 0.0):.4f} | "
                    f"com[pos/na/no/nb] {loss_items.get('com_p', 0.0):.4f}/{loss_items.get('com_na', 0.0):.4f}/"
                    f"{loss_items.get('com_no', 0.0):.4f}/{loss_items.get('com_nb', 0.0):.4f} | "
                    f"com_gap {loss_items.get('com_g1', 0.0):.4f},{loss_items.get('com_g2', 0.0):.4f},{loss_items.get('com_g3', 0.0):.4f} | "
                    f"glb[pos/na/no/nb] {loss_items.get('glb_p', 0.0):.4f}/{loss_items.get('glb_na', 0.0):.4f}/"
                    f"{loss_items.get('glb_no', 0.0):.4f}/{loss_items.get('glb_nb', 0.0):.4f} | "
                    f"glb_gap {loss_items.get('glb_g1', 0.0):.4f},{loss_items.get('glb_g2', 0.0):.4f},{loss_items.get('glb_g3', 0.0):.4f} | "
                    f"att[pos/neg/gap] {loss_items.get('att_p', 0.0):.4f}/{loss_items.get('att_n', 0.0):.4f}/{loss_items.get('att_g', 0.0):.4f} | "
                    f"obj[pos/neg/gap] {loss_items.get('obj_p', 0.0):.4f}/{loss_items.get('obj_n', 0.0):.4f}/{loss_items.get('obj_g', 0.0):.4f}"
                )
        scheduler.step()

        progress_bar.close()
        mean_train_loss = float(np.mean(epoch_train_losses))
        print(f"epoch {epoch + 1} train loss {mean_train_loss}")
        train_losses.append(mean_train_loss)

        print(f"epoch {epoch + 1}: evaluation metrics are being written to {eval_log_path}")
        if args.open_world:
            val_result = test.evaluate_ow(model, val_dataset, args, verbose=False)
            val_result_str = " | ".join([f"{k} {round(v, 4)}" for k, v in val_result.items()])
        else:
            val_result, val_result_str = evaluate(model, val_dataset, args, verbose=False)
        val_results.append(val_result)

        if args.open_world:
            test_result = test.evaluate_ow(model, test_dataset, args, verbose=False)
            test_result_str = " | ".join([f"{k} {round(v, 4)}" for k, v in test_result.items()])
        else:
            test_result, test_result_str = evaluate(model, test_dataset, args, verbose=False)
        _append_eval_log(eval_log_path, f"epoch {epoch + 1} val:  {val_result_str}")
        _append_eval_log(eval_log_path, f"epoch {epoch + 1} test: {test_result_str}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": mean_train_loss,
            "val": {k: float(v) for k, v in val_result.items()},
            "test": {k: float(v) for k, v in test_result.items()},
        }
        history.append(epoch_record)

        epoch_json_path = os.path.join(args.save_path, f"eval_epoch_{epoch + 1}.json")
        with open(epoch_json_path, "w", encoding="utf-8") as f:
            json.dump(epoch_record, f, indent=2, ensure_ascii=False)

        if args.val_metric == 'best_loss' and val_result.get('best_loss', None) is not None:
            if val_result['best_loss'] < best_loss:
                best_loss = val_result['best_loss']
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.save_path, "val_best.pt"))
        if args.val_metric != 'best_loss':
            if val_result[args.val_metric] > best_metric:
                best_metric = val_result[args.val_metric]
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.save_path, "val_best.pt"))

        final_model_state = model.state_dict()
        if epoch + 1 == args.epochs:
            print("--- loading best validation checkpoint for final test evaluation ---")
            model.load_state_dict(torch.load(os.path.join(args.save_path, "val_best.pt")))

    if args.open_world:
        final_test_result = test.evaluate_ow(model, test_dataset, args, verbose=False)
        final_test_result_str = " | ".join([f"{k} {round(v, 4)}" for k, v in final_test_result.items()])
    else:
        final_test_result, final_test_result_str = evaluate(model, test_dataset, args, verbose=False)
    _append_eval_log(eval_log_path, f"final_test: {final_test_result_str}")

    try:
        os.makedirs(args.save_path, exist_ok=True)
    except Exception:
        pass

    best_score = float(best_loss) if args.val_metric == 'best_loss' else float(best_metric)
    summary = {
        "val_metric": args.val_metric,
        "best_epoch": int(best_epoch + 1),
        "best_score": best_score,
        "train_losses": [float(x) for x in train_losses],
        "history": history,
        "final_test": {k: float(v) for k, v in final_test_result.items()},
    }

    json_path = os.path.join(args.save_path, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Evaluation results written to: {json_path}")
    print(f"[INFO] Evaluation log written to: {eval_log_path}")

    if args.save_final_model:
        torch.save(final_model_state, os.path.join(args.save_path, 'final_model.pt'))


if __name__ == '__main__':
    args = parser.parse_args()
    if args.config:
        load_args(args.config, args)
    print(args)

    dataset_path = args.dataset_path
    set_seed(args.seed)

    train_dataset = CompositionDataset(dataset_path, phase='train', split='compositional-split-natural', open_world=args.open_world)
    val_dataset = CompositionDataset(dataset_path, phase='val', split='compositional-split-natural', open_world=args.open_world)
    test_dataset = CompositionDataset(dataset_path, phase='test', split='compositional-split-natural', open_world=args.open_world)

    model, optimizer = configure_model(args, train_dataset)
    model.to(device)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    train_model(model, scheduler, optimizer, args, train_dataset, val_dataset, test_dataset)
