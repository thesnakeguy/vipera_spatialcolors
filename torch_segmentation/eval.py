import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import make_datasets, MEAN, STD
from losses import IoUMeter
from model import build_model


def denorm(x: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization -> HWC float image in [0, 1] for display."""
    img = x.cpu().numpy().transpose(1, 2, 0) * STD + MEAN
    return img.clip(0.0, 1.0)


def per_image_iou_fg(pred: torch.Tensor, target: torch.Tensor) -> float:
    p, t = pred.bool(), target.bool()
    union = (p | t).sum().item()
    if union == 0:
        return 1.0
    return (p & t).sum().item() / union


def _overlay(ax, img, mask, color, img_alpha=0.45, mask_alpha=0.6):
    """Show the (dimmed) original image with a colored mask drawn on top.
    """
    ax.imshow(img, alpha=img_alpha)
    rgba = np.zeros((*mask.shape, 4), np.float32)
    rgba[..., :3] = color
    rgba[..., 3] = mask * mask_alpha
    ax.imshow(rgba)


def save_triptych(img, gt, pred, iou, path: Path, img_alpha=0.45, thr=0.5):
    """[original | original+ground-truth | original+prediction] for one image.
    """
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(img);                              axes[0].set_title("original")
    _overlay(axes[1], img, gt, color=(0.0, 1.0, 0.0), img_alpha=img_alpha);   axes[1].set_title("ground truth")
    _overlay(axes[2], img, pred, color=(1.0, 0.0, 0.0), img_alpha=img_alpha); axes[2].set_title(f"prediction (thr={thr:.2f})")
    fig.suptitle(f"{path.stem}   iou_fg={iou:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_curves(history: list, path: Path):
    if not history:
        return
    x = list(range(len(history)))
    g = lambda k: [h.get(k) for h in history]
    s2 = next((i for i, h in enumerate(history) if h["stage"] == "stage2"), None)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.plot(x, g("train_loss"), label="train_loss")
    a1.plot(x, g("val_loss"), label="val_loss")
    a1.set_title("loss"); a1.set_xlabel("epoch (across stages)"); a1.legend()

    a2.plot(x, g("val_iou_fg"), label="val_iou_fg")
    a2.plot(x, g("val_leakage"), label="val_leakage")
    a2.plot(x, g("val_boundary_iou"), label="val_boundary_iou")
    a2.set_title("validation metrics"); a2.set_xlabel("epoch (across stages)")
    a2.set_ylim(0, 1); a2.legend()

    for ax in (a1, a2):
        if s2 is not None:
            ax.axvline(s2 - 0.5, color="gray", ls="--", lw=1, label="stage2")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def default_grid() -> list[float]:
    """Threshold grid for the sweep: 0.30 .. 0.90 step 0.05."""
    return [round(0.30 + 0.05 * i, 2) for i in range(13)]


def _load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(encoder=ckpt.get("encoder"), weights=None,
                        arch=ckpt.get("arch", "unet")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def sweep_thresholds(ckpt_path: Path, splits: list[str], grid: list[float],
                     workers: int, bs: int = 8, max_batches: int = 0,
                     out_dir: "Path | str | None" = None):
    """One forward pass per split -> metrics at every threshold in `grid`.

    The threshold is a post-hoc operating point: the model emits the same
    sigmoid probabilities regardless, so we binarize once per threshold instead
    of retraining. Writes threshold_sweep.json with {split: {thr: metrics}} so
    run_seeds can calibrate on val and report on test.

    out_dir: where to write threshold_sweep.json. Default (standalone use) is a
    dedicated `<ckpt dir>/eval/` folder; run_seeds passes the run dir explicitly
    to keep its own layout unchanged.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = _load_model(ckpt_path, device)
    seed, size = ckpt.get("seed", 42), ckpt.get("size", 512)
    out = Path(out_dir) if out_dir is not None else ckpt_path.parent / "eval"
    out.mkdir(parents=True, exist_ok=True)
    print(f"sweep ckpt={ckpt_path}  seed={seed}  thresholds={grid}  out={out}")

    result = {"thresholds": grid, "seed": seed, "splits": {}}
    for split in splits:
        ds = make_datasets(size=size, seed=seed)[split]
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            num_workers=workers, pin_memory=True)
        meters = {t: IoUMeter() for t in grid}
        for bi, (x, y) in enumerate(loader):
            if max_batches and bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits = model(x).float()
            for t in grid:
                meters[t].update(logits, y, thr=t)
        result["splits"][split] = {f"{t:.3f}": meters[t].compute() for t in grid}
        m = result["splits"][split]
        print(f"  {split:5s}: " + "  ".join(
            f"t={t:.2f} fg={m[f'{t:.3f}']['iou_fg']:.3f} "
            f"leak={m[f'{t:.3f}']['leakage']:.3f} rec={m[f'{t:.3f}']['recall']:.3f}"
            for t in (grid[0], grid[len(grid)//2], grid[-1])))
    (out / "threshold_sweep.json").write_text(json.dumps(result, indent=2))
    print(f"wrote {out / 'threshold_sweep.json'}")
    return result


@torch.no_grad()
def evaluate(ckpt_path: Path, split: str, save_figs: bool, max_figs: int,
             workers: int, threshold: float = 0.5, img_alpha: float = 0.45):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    seed = ckpt.get("seed", 42)
    size = ckpt.get("size", 512)
    encoder = ckpt.get("encoder")
    arch = ckpt.get("arch", "unet")
    out = ckpt_path.parent
    print(f"ckpt={ckpt_path}  seed={seed}  size={size}  encoder={encoder}  "
          f"arch={arch}  device={device}  threshold={threshold}")

    model = build_model(encoder=encoder, weights=None, arch=arch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = make_datasets(size=size, seed=seed)[split]
    # bs=1, no shuffle -> batch index i maps to ds.ids[i] (stable filenames)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers,
                        pin_memory=True)
    print(f"{split} set: {len(ds)} images")

    # Non-default thresholds write to their own dirs/files so the 0.5 baseline
    # figures + metrics are never clobbered (e.g. preds_thr0.90/, test_metrics_thr0.90.json).
    tag = "" if abs(threshold - 0.5) < 1e-9 else f"_thr{threshold:.2f}"
    fig_dir = out / f"preds{tag}"
    if save_figs:
        fig_dir.mkdir(parents=True, exist_ok=True)

    meter = IoUMeter()
    per_image = []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        meter.update(logits.float(), y, thr=threshold)

        pred = (torch.sigmoid(logits) > threshold).float()
        iou_i = per_image_iou_fg(pred, y)
        stem = Path(ds.images[ds.ids[i]]["file_name"]).stem
        per_image.append({"image": stem, "iou_fg": iou_i})

        if save_figs and (not max_figs or i < max_figs):
            save_triptych(denorm(x[0]), y[0, 0].cpu().numpy(),
                          pred[0, 0].cpu().numpy(), iou_i, fig_dir / f"{stem}.png",
                          img_alpha=img_alpha, thr=threshold)

    metrics = meter.compute()
    print(f"\n{split} metrics (thr={threshold}):")
    for k, v in metrics.items():
        print(f"  {k:13s} {v:.4f}")

    result = {"split": split, "seed": seed, "n": len(ds), "encoder": encoder,
              "arch": arch, "threshold": threshold, "metrics": metrics,
              "per_image": per_image}
    (out / f"{split}_metrics{tag}.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out / f'{split}_metrics{tag}.json'}")

    hist_path = out / "history.json"
    if hist_path.exists():
        plot_curves(json.loads(hist_path.read_text()), out / "curves.png")
        print(f"wrote {out / 'curves.png'}")
    if save_figs:
        n = len(per_image) if not max_figs else min(max_figs, len(per_image))
        print(f"wrote {n} prediction subplots to {fig_dir}/")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to best.pt")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--no-figs", action="store_true",
                   help="skip per-image subplots (metrics + curves only)")
    p.add_argument("--max-figs", type=int, default=0,
                   help="cap number of subplots saved (0 = all)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="decision threshold for metrics + figures (non-sweep). "
                        "!=0.5 writes preds_thr<t>/ and {split}_metrics_thr<t>.json")
    p.add_argument("--alpha", type=float, default=0.45,
                   help="overlay image alpha in triptychs (higher = more photo visible)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sweep", action="store_true",
                   help="evaluate a grid of decision thresholds (one forward "
                        "pass) -> threshold_sweep.json; no figures.")
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                   help="threshold grid for --sweep (default 0.30..0.90 step 0.05)")
    p.add_argument("--sweep-splits", type=str, nargs="+", default=["val", "test"],
                   help="splits to sweep (calibrate on val, report on test)")
    p.add_argument("--sweep-out", type=str, default=None,
                   help="dir for --sweep output (default: <ckpt dir>/eval/)")
    p.add_argument("--bs", type=int, default=8, help="batch size for --sweep")
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap batches/split for --sweep (smoke test)")
    args = p.parse_args()
    if args.sweep:
        sweep_thresholds(Path(args.ckpt), args.sweep_splits,
                         args.thresholds or default_grid(), args.workers,
                         bs=args.bs, max_batches=args.max_batches,
                         out_dir=args.sweep_out)
    else:
        evaluate(Path(args.ckpt), args.split, not args.no_figs, args.max_figs,
                 args.workers, threshold=args.threshold, img_alpha=args.alpha)


if __name__ == "__main__":
    main()
