import torch

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count_table



def prepare_dummy_input(
        device,
        batch_size=1,
        image_size=224
):

    images = torch.randn(
        batch_size,
        3,
        image_size,
        image_size
    ).to(device)


    # according to CAMS forward:
    # img, att_id, obj_id, pair_id

    att_id = torch.randint(
        0,
        100,
        (batch_size,)
    ).to(device)


    obj_id = torch.randint(
        0,
        100,
        (batch_size,)
    ).to(device)


    pair_id = torch.randint(
        0,
        1000,
        (batch_size,)
    ).to(device)


    return [
        images,
        att_id,
        obj_id,
        pair_id
    ]



class InferenceWrapper(torch.nn.Module):

    def __init__(self, model):

        super().__init__()

        self.model = model


    def forward(self, data):

        loss = self.model(data)

        return loss 



def compute_inference_flops(
        model,
        device,
        batch_size=1
):

    model.eval()


    dummy_input = prepare_dummy_input(
        device,
        batch_size
    )


    wrapper = InferenceWrapper(
        model
    ).to(device)


    flops = FlopCountAnalysis(
        wrapper,
        (dummy_input,)
    )


    print("="*60)
    print("Inference FLOPs")
    print("="*60)


    print(
        "Total FLOPs:",
        flops.total()
    )


    print(
        "GFLOPs:",
        flops.total()/1e9
    )


    print(
        parameter_count_table(
            model
        )
    )


    return flops.total()



def compute_training_flops(
        model,
        device,
        batch_size=1
):

    model.train()


    data = prepare_dummy_input(
        device,
        batch_size
    )


    # forward FLOPs

    flops = FlopCountAnalysis(
        model,
        (data,)
    )


    forward_flops = flops.total()


    # backward approximately 2x forward
    # common approximation

    training_flops = (
        forward_flops * 3
    )


    print("="*60)
    print("Training FLOPs")
    print("="*60)


    print(
        "Forward FLOPs:",
        forward_flops
    )


    print(
        "Estimated Training FLOPs:",
        training_flops
    )


    print(
        "Training GFLOPs:",
        training_flops/1e9
    )


    return training_flops



def analyze_flops(
        model,
        device
):

    print("\nFLOPs Analysis Start\n")


    inference_flops = compute_inference_flops(
        model,
        device
    )


    training_flops = compute_training_flops(
        model,
        device
    )


    print("\nSummary")
    print("-"*60)

    print(
        f"Inference: {inference_flops/1e9:.3f} GFLOPs"
    )

    print(
        f"Training: {training_flops/1e9:.3f} GFLOPs"
    )
