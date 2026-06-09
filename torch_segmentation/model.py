import segmentation_models_pytorch as smp
import torch.nn as nn

ENCODER = "timm-efficientnet-b0"
ARCHS = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus}


def build_model(encoder: str = ENCODER, weights: str | None = "imagenet",
                classes: int = 1, arch: str = "unet"):
    return ARCHS[arch](
        encoder_name=encoder,
        encoder_weights=weights,
        in_channels=3,
        classes=classes,
    )


def set_encoder_trainable(model, trainable: bool) -> None:
    """Freeze (False) or unfreeze (True) the whole encoder."""
    for p in model.encoder.parameters():
        p.requires_grad = trainable


def freeze_frozen_batchnorm(model) -> None:
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(
                p.requires_grad for p in m.parameters(recurse=False)):
            m.eval()


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


if __name__ == "__main__":
    m = build_model()
    tr, tot = count_params(m)
    print(f"all trainable      : {tr/1e6:.2f}M / {tot/1e6:.2f}M")
    set_encoder_trainable(m, False)
    tr, tot = count_params(m)
    print(f"encoder frozen      : {tr/1e6:.2f}M / {tot/1e6:.2f}M trainable (decoder only)")
    set_encoder_trainable(m, True)
    tr, tot = count_params(m)
    print(f"encoder unfrozen    : {tr/1e6:.2f}M / {tot/1e6:.2f}M trainable")
