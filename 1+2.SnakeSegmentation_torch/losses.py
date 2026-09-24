import torch
import torch.nn.functional as F


def dice_loss(logits, target, smooth: float = 1.0):
    prob = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    inter = (prob * target).sum(1)
    denom = prob.sum(1) + target.sum(1)
    dice = (2 * inter + smooth) / (denom + smooth)
    return 1 - dice.mean()


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return 0.5 * dice_loss(logits, target) + 0.5 * bce


def focal_tversky_loss(logits, target, alpha: float = 0.7, beta: float = 0.3,
                       gamma: float = 1.333, smooth: float = 1.0):
    """Focal-Tversky loss -- anti-leakage variant.

    Tversky index = TP / (TP + alpha*FP + beta*FN). With alpha > beta, false
    positives (background predicted as snake = leakage) are penalized harder
    than false negatives, which is exactly what the downstream color analysis
    wants. The focal exponent gamma>1 concentrates learning on hard examples.
    we might need this for the color study!!
    """
    prob = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    tp = (prob * target).sum(1)
    fp = (prob * (1 - target)).sum(1)
    fn = ((1 - prob) * target).sum(1)
    ti = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return ((1 - ti).clamp(min=0) ** gamma).mean()


def get_loss(name: str = "dicebce"):
    losses = {"dicebce": combined_loss, "focal_tversky": focal_tversky_loss}
    if name not in losses:
        raise ValueError(f"unknown loss {name!r}; choose from {list(losses)}")
    return losses[name]


def _erode(mask, d: int):
    """Binary erosion by a (2d+1) square, via min-pool = 1 - maxpool(1 - x)."""
    k = 2 * d + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=d)


class IoUMeter:
    """Accumulate TP/FP/FN/TN (and boundary IoU) across batches; compute at end.
    """

    def __init__(self, dilation_ratio: float = 0.02):
        self.tp = self.fp = self.fn = self.tn = 0.0
        self.i_bnd = self.u_bnd = 0.0
        self.dilation_ratio = dilation_ratio

    @torch.no_grad()
    def update(self, logits, target, thr: float = 0.5):
        pred = (torch.sigmoid(logits) > thr).float()
        t = target
        self.tp += (pred * t).sum().item()
        self.fp += (pred * (1 - t)).sum().item()
        self.fn += ((1 - pred) * t).sum().item()
        self.tn += ((1 - pred) * (1 - t)).sum().item()
        # boundary band = mask minus its erosion; IoU of the two edge bands
        h, w = t.shape[-2:]
        d = max(1, round(self.dilation_ratio * (h ** 2 + w ** 2) ** 0.5))
        gb, pb = t - _erode(t, d), pred - _erode(pred, d)
        self.i_bnd += (gb * pb).sum().item()
        self.u_bnd += (gb + pb - gb * pb).sum().item()

    def compute(self) -> dict:
        eps = 1e-6
        iou_fg = self.tp / max(self.tp + self.fp + self.fn, eps)
        iou_bg = self.tn / max(self.tn + self.fp + self.fn, eps)
        precision = self.tp / max(self.tp + self.fp, eps)
        recall = self.tp / max(self.tp + self.fn, eps)
        return {
            "iou_fg": iou_fg,
            "leakage": 1.0 - precision,
            "recall": recall,
            "boundary_iou": self.i_bnd / max(self.u_bnd, eps),
            "iou_bg": iou_bg,
            "iou_mean": 0.5 * (iou_fg + iou_bg),
        }


if __name__ == "__main__":
    # sanity: perfect prediction -> loss ~0, IoU 1.0
    target = (torch.rand(2, 1, 64, 64) > 0.5).float()
    logits_perfect = (target * 2 - 1) * 20  # large +/- logits matching target
    print(f"perfect  | loss={combined_loss(logits_perfect, target):.4f}")
    m = IoUMeter(); m.update(logits_perfect, target); print("         |", m.compute())

    logits_bad = -logits_perfect  # fully wrong
    print(f"inverted | loss={combined_loss(logits_bad, target):.4f}")
    m = IoUMeter(); m.update(logits_bad, target); print("         |", m.compute())
