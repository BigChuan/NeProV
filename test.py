
import copy
import torch
import numpy as np
from utils import *
from tqdm import tqdm
from flags import parser
from scipy.stats import hmean
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataset import CompositionDataset
from torch.utils.data.dataloader import DataLoader

cudnn.benchmark = True

device = "cuda" if torch.cuda.is_available() else "cpu"

# Evaluation class. See "Learning Graph Embeddings for Open World Compositional Zero-Shot Learning"
class Evaluator:
    """
    Evaluator class, adapted from:
    https://github.com/Tushar-N/attributes-as-operators

    With modifications from:
    https://github.com/ExplainableML/czsl
    """

    def __init__(self, dset, verbose=True):

        self.verbose = verbose

        self.dset = dset

        # Convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe',
        # 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                 for attr, obj in dset.pairs]
        self.train_pairs = [(dset.attr2idx[attr], dset.obj2idx[obj])
                            for attr, obj in dset.train_pairs]
        self.pairs = torch.LongTensor(pairs)

        # Mask over pairs that occur in closed world
        # Select set based on phase
        if dset.phase == 'train':
            if self.verbose:
                print('Evaluating with train pairs')
            test_pair_set = set(dset.train_pairs)
            test_pair_gt = set(dset.train_pairs)
        elif dset.phase == 'val':
            if self.verbose:
                print('Evaluating with validation pairs')
            test_pair_set = set(dset.val_pairs + dset.train_pairs)
            test_pair_gt = set(dset.val_pairs)
        else:
            if self.verbose:
                print('Evaluating with test pairs')
            test_pair_set = set(dset.test_pairs + dset.train_pairs)
            test_pair_gt = set(dset.test_pairs)

        self.test_pair_dict = [
            (dset.attr2idx[attr],
             dset.obj2idx[obj]) for attr,
            obj in test_pair_gt]
        self.test_pair_dict = dict.fromkeys(self.test_pair_dict, 0)

        # dict values are pair val, score, total
        for attr, obj in test_pair_gt:
            pair_val = dset.pair2idx[(attr, obj)]
            key = (dset.attr2idx[attr], dset.obj2idx[obj])
            self.test_pair_dict[key] = [pair_val, 0, 0]

        # open world
        if dset.open_world:
            masks = [1 for _ in dset.pairs]
        else:
            masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        # masks = [1 if pair in test_pair_set else 0 for pair in dset.pairs]

        self.closed_mask = torch.BoolTensor(masks)
        # Mask of seen concepts
        seen_pair_set = set(dset.train_pairs)
        mask = [1 if pair in seen_pair_set else 0 for pair in dset.pairs]
        self.seen_mask = torch.BoolTensor(mask)

        # Object specific mask over which pairs occur in the object oracle
        # setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj == obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.BoolTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # Decide if the model under evaluation is a manifold model or not
        self.score_model = self.score_manifold_model

    # Generate mask for each settings, mask scores, and get prediction labels
    def generate_predictions(self, scores, obj_truth, bias=0.0, topk=1):  # (Batch, #pairs)
        '''
        Inputs
            scores: Output scores
            obj_truth: Ground truth object
        Returns
            results: dict of results in 3 settings
        '''

        def get_pred_from_scores(_scores, topk):
            """
            Given list of scores, returns top 10 attr and obj predictions
            Check later
            """
            _, pair_pred = _scores.topk(
                topk, dim=1)  # sort returns indices of k largest values
            pair_pred = pair_pred.contiguous().view(-1)
            attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(
                -1, topk
            ), self.pairs[pair_pred][:, 1].view(-1, topk)
            return (attr_pred, obj_pred)

        results = {}
        orig_scores = scores.clone()
        mask = self.seen_mask.repeat(
            scores.shape[0], 1
        )  # Repeat mask along pairs dimension
        scores[~mask] += bias  # Add bias to test pairs

        # Unbiased setting

        # Open world setting --no mask, all pairs of the dataset
        results.update({"open": get_pred_from_scores(scores, topk)})
        results.update(
            {"unbiased_open": get_pred_from_scores(orig_scores, topk)}
        )
        # Closed world setting - set the score for all Non test pairs to -1e10,
        # this excludes the pairs from set not in evaluation
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10
        closed_orig_scores = orig_scores.clone()
        closed_orig_scores[~mask] = -1e10
        results.update({"closed": get_pred_from_scores(closed_scores, topk)})
        results.update(
            {"unbiased_closed": get_pred_from_scores(closed_orig_scores, topk)}
        )

        return results

    def score_clf_model(self, scores, obj_truth, topk=1):
        '''
        Wrapper function to call generate_predictions for CLF models
        '''
        attr_pred, obj_pred = scores

        # Go to CPU
        attr_pred, obj_pred, obj_truth = attr_pred.to(
            'cpu'), obj_pred.to('cpu'), obj_truth.to('cpu')

        # Gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # Multiply P(a) * P(o) to get P(pair)
        # Return only attributes that are in our pairs
        attr_subset = attr_pred.index_select(1, self.pairs[:, 0])
        obj_subset = obj_pred.index_select(1, self.pairs[:, 1])
        scores = (attr_subset * obj_subset)  # (Batch, #pairs)

        results = self.generate_predictions(scores, obj_truth)
        results['biased_scores'] = scores

        return results

    def score_manifold_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''
        # Go to CPU
        scores = {k: v.to('cpu') for k, v in scores.items()}
        obj_truth = obj_truth.to(device)

        # Gather scores for all relevant (a,o) pairs
        scores = torch.stack(
            [scores[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)
        orig_scores = scores.clone()
        results = self.generate_predictions(scores, obj_truth, bias, topk)
        results['scores'] = orig_scores
        return results

    def score_fast_model(self, scores, obj_truth, bias=0.0, topk=1):
        '''
        Wrapper function to call generate_predictions for manifold models
        '''

        results = {}
        # Repeat mask along pairs dimension
        mask = self.seen_mask.repeat(scores.shape[0], 1)
        scores[~mask] += bias  # Add bias to test pairs

        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[~mask] = -1e10

        # sort returns indices of k largest values
        _, pair_pred = closed_scores.topk(topk, dim=1)
        # _, pair_pred = scores.topk(topk, dim=1)  # sort returns indices of k
        # largest values
        pair_pred = pair_pred.contiguous().view(-1)
        attr_pred, obj_pred = self.pairs[pair_pred][:, 0].view(-1, topk), \
            self.pairs[pair_pred][:, 1].view(-1, topk)

        results.update({'closed': (attr_pred, obj_pred)})
        return results

    def evaluate_predictions(
            self,
            predictions,
            attr_truth,
            obj_truth,
            pair_truth,
            allpred,
            topk=1):
        # Go to CPU
        attr_truth, obj_truth, pair_truth = (
            attr_truth.to("cpu"),
            obj_truth.to("cpu"),
            pair_truth.to("cpu"),
        )

        pairs = list(zip(list(attr_truth.numpy()), list(obj_truth.numpy())))

        seen_ind, unseen_ind = [], []
        for i in range(len(attr_truth)):
            if pairs[i] in self.train_pairs:
                seen_ind.append(i)
            else:
                unseen_ind.append(i)

        seen_ind, unseen_ind = torch.LongTensor(seen_ind), torch.LongTensor(
            unseen_ind
        )

        def _process(_scores):
            # Top k pair accuracy
            # Attribute, object and pair
            attr_match = (
                attr_truth.unsqueeze(1).repeat(1, topk) == _scores[0][:, :topk]
            )
            obj_match = (
                obj_truth.unsqueeze(1).repeat(1, topk) == _scores[1][:, :topk]
            )

            # Match of object pair
            match = (attr_match * obj_match).any(1).float()
            attr_match = attr_match.any(1).float()
            obj_match = obj_match.any(1).float()
            # Match of seen and unseen pairs
            seen_match = match[seen_ind]
            unseen_match = match[unseen_ind]
            # Calculating class average accuracy

            seen_score, unseen_score = torch.ones(512, 5), torch.ones(512, 5)

            return attr_match, obj_match, match, seen_match, unseen_match, torch.Tensor(
                seen_score + unseen_score), torch.Tensor(seen_score), torch.Tensor(unseen_score)

        def _add_to_dict(_scores, type_name, stats):
            base = [
                "_attr_match",
                "_obj_match",
                "_match",
                "_seen_match",
                "_unseen_match",
                "_ca",
                "_seen_ca",
                "_unseen_ca",
            ]
            for val, name in zip(_scores, base):
                stats[type_name + name] = val

        stats = dict()

        # Closed world
        closed_scores = _process(predictions["closed"])
        unbiased_closed = _process(predictions["unbiased_closed"])
        _add_to_dict(closed_scores, "closed", stats)
        _add_to_dict(unbiased_closed, "closed_ub", stats)

        # Calculating AUC
        scores = predictions["scores"]
        # getting score for each ground truth class
        correct_scores = scores[torch.arange(scores.shape[0]), pair_truth][
            unseen_ind
        ]

        # Getting top predicted score for these unseen classes
        max_seen_scores = predictions['scores'][unseen_ind][:, self.seen_mask].topk(topk, dim=1)[
            0][:, topk - 1]

        # Getting difference between these scores
        unseen_score_diff = max_seen_scores - correct_scores

        # Getting matched classes at max bias for diff
        unseen_matches = stats["closed_unseen_match"].bool()
        correct_unseen_score_diff = unseen_score_diff[unseen_matches] - 1e-4

        # sorting these diffs
        correct_unseen_score_diff = torch.sort(correct_unseen_score_diff)[0]
        magic_binsize = 20
        # getting step size for these bias values
        bias_skip = max(len(correct_unseen_score_diff) // magic_binsize, 1)
        # Getting list
        biaslist = correct_unseen_score_diff[::bias_skip]

        seen_match_max = float(stats["closed_seen_match"].mean())
        unseen_match_max = float(stats["closed_unseen_match"].mean())
        seen_accuracy, unseen_accuracy = [], []

        # Go to CPU
        base_scores = {k: v.to("cpu") for k, v in allpred.items()}
        obj_truth = obj_truth.to("cpu")

        # Gather scores for all relevant (a,o) pairs
        base_scores = torch.stack(
            [allpred[(attr, obj)] for attr, obj in self.dset.pairs], 1
        )  # (Batch, #pairs)

        for bias in biaslist:
            scores = base_scores.clone()
            results = self.score_fast_model(
                scores, obj_truth, bias=bias, topk=topk)
            results = results['closed']  # we only need biased
            results = _process(results)
            seen_match = float(results[3].mean())
            unseen_match = float(results[4].mean())
            seen_accuracy.append(seen_match)
            unseen_accuracy.append(unseen_match)

        seen_accuracy.append(seen_match_max)
        unseen_accuracy.append(unseen_match_max)
        seen_accuracy, unseen_accuracy = np.array(seen_accuracy), np.array(
            unseen_accuracy
        )
        area = np.trapezoid(seen_accuracy, unseen_accuracy)

        for key in stats:
            stats[key] = float(stats[key].mean())

        try:
            harmonic_mean = hmean([seen_accuracy, unseen_accuracy], axis=0)
        except BaseException:
            harmonic_mean = 0

        max_hm = np.max(harmonic_mean)
        idx = np.argmax(harmonic_mean)
        if idx == len(biaslist):
            bias_term = 1e3
        else:
            bias_term = biaslist[idx]
        stats["biasterm"] = float(bias_term)
        stats["best_unseen"] = np.max(unseen_accuracy)
        stats["best_seen"] = np.max(seen_accuracy)
        stats["AUC"] = area
        stats["hm_unseen"] = unseen_accuracy[idx]
        stats["hm_seen"] = seen_accuracy[idx]
        stats["best_hm"] = max_hm
        return stats




def predict_logits(model, dataset, args):
    model.eval()
    device = next(model.parameters()).device

    dataloader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )

    # ------------------------------------------------------------
    # 1. 准备 pair 索引（所有组合）
    # ------------------------------------------------------------
    obj2idx = dataset.obj2idx
    attr2idx = dataset.attr2idx
    pairs = torch.tensor(
        [(attr2idx[attr], obj2idx[obj]) for attr, obj in dataset.pairs],
        dtype=torch.long,
        device=device,
    )  # [K,2], K = #pairs

    # ------------------------------------------------------------
    # 2. 先算 attr / obj 的 text prompts（与 pair 无关，只用一次）
    # ------------------------------------------------------------
    with torch.no_grad():
        # dummy pair，只是为了走 construct_token_tensors，真正只用 attr/obj 分支
        dummy_pairs = torch.zeros(1, 2, dtype=torch.long, device=device)
        token_dict_ao = model.csp.construct_token_tensors(dummy_pairs)

        # attr / obj token tensors
        attr_pos_tokens = token_dict_ao["attr_pos"]  # [num_attr, L, D]
        attr_neg_tokens = token_dict_ao["attr_neg"]  # [num_attr, L, D]
        obj_pos_tokens  = token_dict_ao["obj_pos"]   # [num_obj, L, D]
        obj_neg_tokens  = token_dict_ao["obj_neg"]   # [num_obj, L, D]

        # text encoder：注意 token_ids 的索引：0->comp, 1->attr, 2->obj
        attr_pos_feats = model.text_encoder(
            model.csp.token_ids[1].to(device),
            attr_pos_tokens,
            enable_pos_emb=True
        )
        attr_neg_feats = model.text_encoder(
            model.csp.token_ids[1].to(device),
            attr_neg_tokens,
            enable_pos_emb=True
        )
        obj_pos_feats = model.text_encoder(
            model.csp.token_ids[2].to(device),
            obj_pos_tokens,
            enable_pos_emb=True
        )
        obj_neg_feats = model.text_encoder(
            model.csp.token_ids[2].to(device),
            obj_neg_tokens,
            enable_pos_emb=True
        )

        # 归一化
        attr_prompt_pos = F.normalize(attr_pos_feats, p=2, dim=-1)  # [Na, D]
        attr_prompt_neg = F.normalize(attr_neg_feats, p=2, dim=-1)
        obj_prompt_pos  = F.normalize(obj_pos_feats,  p=2, dim=-1)  # [No, D]
        obj_prompt_neg  = F.normalize(obj_neg_feats,  p=2, dim=-1)

    # ------------------------------------------------------------
    # 3. 分块计算 comp 的 4 组 prompt features（防止显存炸）
    # ------------------------------------------------------------
    with torch.no_grad():
        chunk_size = getattr(args, "prompt_chunk_size", 128)
        pair_chunks = torch.split(pairs, chunk_size, dim=0)
        com_prompt_reps = torch.Tensor().cuda()
        com_prompt_reps_n = torch.Tensor().cuda()
        
        for batch_pairs in pair_chunks:
            token_dict = model.csp.construct_token_tensors(batch_pairs)

            comp_pos_tokens  = token_dict["comp_pos"]   # [B', L, D]

            com_pos_feats = model.text_encoder(
                model.csp.token_ids[0].to(device),
                comp_pos_tokens,
                enable_pos_emb=True
            )

            com_prompt_rep = F.normalize(com_pos_feats, p=2, dim=-1)  # [B, D]
            # 拼成 [K, D]，顺序与 dataset.pairs 一致
            com_prompt_reps = torch.cat([com_prompt_reps, com_prompt_rep], dim=0)

    # ------------------------------------------------------------
    # 4. 评估所有图像，计算 logits
    # ------------------------------------------------------------
    all_logits = torch.Tensor()
    all_attr_gt, all_obj_gt, all_pair_gt = [], [], []

    # 多负样本权重（如果没在 args 里配，就默认 1.0）

    for idx, data in tqdm(
        enumerate(dataloader), total=len(dataloader), desc="Testing"
    ):
        with torch.no_grad():
            data = [d.to(device) if not isinstance(d, tuple) else d for d in data]
            img   = data[0]
            att_id, obj_id, pair_id = data[1], data[2], data[3]

            # image encoder: 得到四个语义空间
            att, obj, com, glb = model.image_encoder(img)

            att_semantic_reps = F.normalize(att, p=2, dim=-1)
            obj_semantic_reps = F.normalize(obj, p=2, dim=-1)
            com_semantic_reps = F.normalize(com, p=2, dim=-1)
            glb_semantic_reps = F.normalize(glb, p=2, dim=-1)

            # --------- attr / obj：pos - neg --------- #
            att_score = model.temp_logit * att_semantic_reps @ attr_prompt_pos.T \
                - model.temp_logit * att_semantic_reps @ attr_prompt_neg.T
              # [B, Na]

            obj_score = model.temp_logit * obj_semantic_reps @ obj_prompt_pos.T \
                - model.temp_logit * obj_semantic_reps @ obj_prompt_neg.T  # [B, No]

            # attr / obj → pair 级别（和之前一样用 pairs2attr_idx / pairs2obj_idx）
            att_prob = torch.softmax(att_score, dim=-1)[:, dataset.pairs2attr_idx]
            obj_prob = torch.softmax(obj_score, dim=-1)[:, dataset.pairs2obj_idx]
            
            # 2) glb 与 attr/obj 负样本的相似度
            glb_pos_log = model.temp_logit * glb_semantic_reps @ com_prompt_reps.T

            # glb_pos_att = model.temp_logit * glb_semantic_reps @ attr_prompt_pos.T   # [B, A]
            # glb_pos_obj = model.temp_logit * glb_semantic_reps @ obj_prompt_pos.T   # [B, O]
            # glb_neg_att = model.neg_temp_logit * glb_semantic_reps @ attr_prompt_neg.T   # [B, A]
            # glb_neg_obj = model.neg_temp_logit * glb_semantic_reps @ obj_prompt_neg.T   # [B, O]

            # 3) 映射到 pair 维度
            pairs2attr = torch.as_tensor(
                dataset.pairs2attr_idx, device=model.device, dtype=torch.long
            )  # [K]
            pairs2obj = torch.as_tensor(
                dataset.pairs2obj_idx, device=model.device, dtype=torch.long
            )  # [K]

            # 从 attr / obj logit 里挑出每个 pair 对应的 attr / obj logit → [B, K]
            # glb_att = (glb_pos_att - glb_neg_att)[:, pairs2attr]  # [B, K]
            # glb_obj = (glb_pos_obj - glb_neg_obj)[:, pairs2obj]   # [B, K]
            
            # 5) 最终 glb branch
            glb_score = glb_pos_log # [B, K]
            
            # --------- comp：pos - neg --------- #
            # 1) comp 与 comp 正样本的相似度: [B, K]
            # 2) com 与 attr/obj 负样本的相似度
            com_pos_log = model.temp_logit * com_semantic_reps @ com_prompt_reps.T

            # 3) 映射到 pair 维度
            pairs2attr = torch.as_tensor(
                dataset.pairs2attr_idx, device=model.device, dtype=torch.long
            )  # [K]
            pairs2obj = torch.as_tensor(
                dataset.pairs2obj_idx, device=model.device, dtype=torch.long
            )  # [K]

            # 从 attr / obj logit 里挑出每个 pair 对应的 attr / obj logit → [B, K]
            # com_att = (com_pos_att - com_neg_att)[:, pairs2attr]  # [B, K]
            # com_obj = (com_pos_obj - com_neg_obj)[:, pairs2obj]   # [B, K]
            # 4) 合成一个负项（这里简单取平均，你也可以加权）
            com_score = com_pos_log  # [B, K]

            # 最终组合 logits（跟原来一样：beta * glb + (1-beta)*com + attr*obj）
            com_final = (
                args.beta * glb_score
                + (1 - args.beta) * com_score
                + att_prob * obj_prob
            )  # [B, K]

            all_logits = torch.cat([all_logits, com_final.cpu()], dim=0)
            all_attr_gt.append(att_id.cpu())
            all_obj_gt.append(obj_id.cpu())
            all_pair_gt.append(pair_id.cpu())

    all_attr_gt = torch.cat(all_attr_gt)
    all_obj_gt  = torch.cat(all_obj_gt)
    all_pair_gt = torch.cat(all_pair_gt)

    return all_logits, all_attr_gt, all_obj_gt, all_pair_gt


def threshold_with_feasibility(
        logits,
        seen_mask,
        threshold=None,
        feasiblity=None):
    """Function to remove infeasible compositions.

    Args:
        logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        seen_mask (torch.tensor): the seen mask with binary
        threshold (float, optional): the threshold value.
            Defaults to None.
        feasiblity (torch.Tensor, optional): the feasibility.
            Defaults to None.

    Returns:
        torch.Tensor: the logits after filtering out the
            infeasible compositions.
    """
    score = copy.deepcopy(logits)
    # Note: Pairs are already aligned here
    mask = (feasiblity >= threshold).float()
    # score = score*mask + (1.-mask)*(-1.)
    score = score * (mask + seen_mask)

    return score

def test(
        test_dataset,
        evaluator,
        all_logits,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt):
    """Function computes accuracy on the validation and
    test dataset.

    Args:
        test_dataset (CompositionDataset): the validation/test
            dataset
        evaluator (Evaluator): the evaluator object
        all_logits (torch.Tensor): the cosine similarities between
            the images and the attribute-object pairs.
        all_attr_gt (torch.tensor): the attribute ground truth
        all_obj_gt (torch.tensor): the object ground truth
        all_pair_gt (torch.tensor): the attribute-object pair ground
            truth
        config (argparse.ArgumentParser): the config

    Returns:
        dict: the result with all the metrics
    """
    """此功能在验证和测试数据集上计算准确性。

    参数：
        test_dataset (CompositionDataset): 验证/测试数据集
        evaluator (Evaluator): 评估器对象
        all_logits (torch.Tensor): 图像与属性-对象对之间的余弦相似度。
        all_attr_gt (torch.tensor): 属性的真实值
        all_obj_gt (torch.tensor): 对象的真实值
        all_pair_gt (torch.tensor): 属性-对象对的真实值
        config (argparse.ArgumentParser): 配置参数

    返回：
        dict: 所有度量的结果
    """

    predictions = {
        pair_name: all_logits[:, i]
        for i, pair_name in enumerate(test_dataset.pairs)
    }
    all_pred = [predictions]

    all_pred_dict = {}
    for k in all_pred[0].keys():
        all_pred_dict[k] = torch.cat(
            [all_pred[i][k] for i in range(len(all_pred))]
        ).float()

    results = evaluator.score_model(
        all_pred_dict, all_obj_gt, bias=1e3, topk=1
    )

    results['predicted_attributes'] = results['unbiased_closed'][0].squeeze(-1).tolist()
    results['predicted_objects'] = results['unbiased_closed'][1].squeeze(-1).tolist()
    results['true_attributes'] = all_attr_gt.tolist()
    results['true_objects'] = all_obj_gt.tolist()

    results['true_pairs'] = all_pair_gt.tolist()


    attr_acc = float(torch.mean(
        (results['unbiased_closed'][0].squeeze(-1) == all_attr_gt).float()))
    obj_acc = float(torch.mean(
        (results['unbiased_closed'][1].squeeze(-1) == all_obj_gt).float()))

    stats = evaluator.evaluate_predictions(
        results,
        all_attr_gt,
        all_obj_gt,
        all_pair_gt,
        all_pred_dict,
        topk=1,
    )

    stats['attr_acc'] = attr_acc
    stats['obj_acc'] = obj_acc

    return stats

def evaluate_ow(model, test_dataset, args, verbose=False):

    predict_logits_func = predict_logits
    feasibility_path = args.feasibility
    unseen_scores = torch.load(
        feasibility_path,
        map_location='cpu',weights_only=True)['feasibility']
    
    with torch.no_grad():
        evaluator = Evaluator(test_dataset, verbose=verbose)
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(
            model, test_dataset,args)
        all_logits = threshold_with_feasibility(
            all_logits,
            test_dataset.seen_mask,
            threshold=args.threshold,
            feasiblity=unseen_scores)
        test_stats = test(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt
        )

    test_saved_results = dict()
    result = ""
    key_set = ["best_seen", "best_unseen", "best_hm", "AUC", "attr_acc", "obj_acc"]
    for key in key_set:
        result = result + key + "  " + str(round(test_stats[key], 4)) + "| "
        test_saved_results[key] = round(test_stats[key], 4)
    if verbose:
        print(result)
    return test_saved_results
        # result = ""
        # for key in test_stats:
        #     result = result + key + "  " + \
        #         str(round(test_stats[key], 4)) + "| "
        # print(result)
        
        
        
from model.configure_model import configure_model

if __name__ == "__main__":

    args = parser.parse_args()
    set_seed(args.seed)
    if args.config:
        load_args(args.config, args)

    print("----")
    test_type = 'OPEN WORLD' if args.open_world else 'CLOSED WORLD'
    print(f"{test_type} evaluation details")
    print("----")
    print(f"dataset: {args.dataset}")
        
    dataset_path = args.dataset_path

    test_dataset = CompositionDataset(dataset_path,
                                       phase='test',
                                       split='compositional-split-natural',
                                       open_world=args.open_world)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False
    )
    model, optimizer = configure_model(args,test_dataset)
    model.to(device)
    
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model,weights_only=True))

    predict_logits_func = predict_logits   
    feasibility_path = args.feasibility
    unseen_scores = torch.load(
        feasibility_path,
        map_location='cpu',weights_only=True)['feasibility']
    with torch.no_grad():
        evaluator = Evaluator(test_dataset, verbose=True)
        all_logits, all_attr_gt, all_obj_gt, all_pair_gt = predict_logits_func(
            model, test_dataset,args)
        all_logits = threshold_with_feasibility(
            all_logits,
            test_dataset.seen_mask,
            threshold=args.threshold,
            feasiblity=unseen_scores)
        test_stats = test(
            test_dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt
        )

        result = ""
        for key in test_stats:
            result = result + key + "  " + \
                str(round(test_stats[key], 4)) + "| "
        print(result)

    print("done!")
    