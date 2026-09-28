import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F


def cosine_dependency(x,y):

    x=torch.tensor(x)
    y=torch.tensor(y)


    x=F.normalize(
        x,
        dim=-1
    )

    y=F.normalize(
        y,
        dim=-1
    )


    sim=(x*y).sum(-1)


    return sim.mean().item()



def linear_probe(
        train_x,
        train_y,
        test_x,
        test_y):


    clf = make_pipeline(
        StandardScaler(),

        LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        )
    )


    clf.fit(
        train_x,
        train_y
    )


    pred = clf.predict(
        test_x
    )


    return accuracy_score(
        test_y,
        pred
    )


def _move_batch_to_device(batch, device):

    moved = []

    for item in batch:

        if torch.is_tensor(item):

            moved.append(
                item.to(device)
            )

        else:
            moved.append(item)

    return moved


@torch.no_grad()
def collect_features(model, loader, device):

    model.eval()

    attr_feats = []
    obj_feats = []
    comp_feats = []

    attrs = []
    objs = []
    comps = []


    for idx, batch in enumerate(loader):
        batch = _move_batch_to_device(batch, device)
        outputs = model(batch, return_features=True)


        attr_feats.append(
            outputs["att_feat"].cpu()
        )

        obj_feats.append(
            outputs["obj_feat"].cpu()
        )

        comp_feats.append(
            outputs["com_feat"].cpu()
        )


        attrs.append(
            batch[1].cpu()
        )

        objs.append(
            batch[2].cpu()
        )

        comps.append(
            batch[3].cpu()
        )


    return {

        "attr_feat":
            torch.cat(attr_feats).numpy(),

        "obj_feat":
            torch.cat(obj_feats).numpy(),

        "com_feat":
            torch.cat(comp_feats).numpy(),

        "attr":
            torch.cat(attrs).numpy(),

        "obj":
            torch.cat(objs).numpy(),

        "comp":
            torch.cat(comps).numpy()
    }
    
    
def disentanglement_analysis(
        model,
        train_loader,
        test_loader,
        device
):
    """
    Representation disentanglement analysis.

    Includes:
        1. Linear probing:
            - attr_feat -> attr
            - attr_feat -> obj
            - obj_feat  -> obj
            - obj_feat  -> attr

        2. Feature dependency:
            - cosine similarity between branches
    """

    print("=" * 70)
    print("Collecting branch representations...")
    print("=" * 70)


    train_feat = collect_features(
        model,
        train_loader,
        device
    )

    test_feat = collect_features(
        model,
        test_loader,
        device
    )


    # =====================================================
    # Linear probing
    # =====================================================

    print("\n" + "=" * 70)
    print("Linear Probe Analysis")
    print("=" * 70)


    results = {}


    # attribute branch
    results["AttrFeat -> Attr"] = linear_probe(
        train_feat["attr_feat"],
        train_feat["attr"],

        test_feat["attr_feat"],
        test_feat["attr"]
    )


    results["AttrFeat -> Obj"] = linear_probe(
        train_feat["attr_feat"],
        train_feat["obj"],

        test_feat["attr_feat"],
        test_feat["obj"]
    )


    # object branch

    results["ObjFeat -> Obj"] = linear_probe(
        train_feat["obj_feat"],
        train_feat["obj"],

        test_feat["obj_feat"],
        test_feat["obj"]
    )


    results["ObjFeat -> Attr"] = linear_probe(
        train_feat["obj_feat"],
        train_feat["attr"],

        test_feat["obj_feat"],
        test_feat["attr"]
    )


    for k,v in results.items():

        print(
            f"{k:<25s}: {v*100:.2f}%"
        )


    # =====================================================
    # Feature dependency
    # =====================================================

    print("\n" + "=" * 70)
    print("Feature Dependency Analysis")
    print("=" * 70)


    dependency = {}


    dependency["Attr-Obj"] = cosine_dependency(
        test_feat["attr_feat"],
        test_feat["obj_feat"]
    )


    dependency["Attr-Comp"] = cosine_dependency(
        test_feat["attr_feat"],
        test_feat["com_feat"]
    )


    dependency["Obj-Comp"] = cosine_dependency(
        test_feat["obj_feat"],
        test_feat["com_feat"]
    )


    for k,v in dependency.items():

        print(
            f"{k:<25s}: {v:.4f}"
        )


    # =====================================================
    # Return all results for saving
    # =====================================================

    return {
        "linear_probe": results,
        "dependency": dependency
    }