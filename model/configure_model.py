import torch
from model.nep import NEP


def configure_model(args, dataset):
    nep = NEP(args, dataset)

    keywords = [
        "csp",
        "ln_norm",
        "tr_",              # visual tr_a/tr_o/tr_c
        "adapter",
        "proj_att",
        "proj_obj",
        "proj_com",
        "temp_logit",
        "cross_attention",
        "latent_",
        "ln_latent",
    ]

    for name, param in nep.named_parameters():
        param.requires_grad = False

    for name, param in nep.named_parameters():
        if any(kw in name for kw in keywords):
            param.requires_grad = True

    total_params = 0
    trainable_params = 0

    print("\n===== Trainable Parameters AFTER configure_model() =====")
    for name, p in nep.named_parameters():
        n = p.numel()
        total_params += n
        if p.requires_grad:
            trainable_params += n
            print(f"[TRAINABLE] {name:60s} {tuple(p.shape)}  numel={n}")
    print("--------------------------------------------------------")
    print(f"Total trainable params : {trainable_params:,}")
    print(f"Total params (all)     : {total_params:,}")
    print("========================================================\n")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, nep.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    return nep, optimizer
