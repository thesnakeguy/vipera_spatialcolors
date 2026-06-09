"""Run identity + provenance metadata.

Every training run gets a unique folder named by `make_run_id` and a
`run_meta.json` capturing *where the results came from*: hyperparameters, the
exact model/loss/augmentation config, the data split (counts + the actual image
ids, so a split is fully reproducible), the software environment, and the git
commit. Written once at start (status "running") so a crashed run still leaves a
trail, then finalized with results + duration at the end.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def make_run_id(seed: int) -> str:
    """Unique, human-readable: <timestamp>-seed<seed>-<githash>."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-seed{seed}-{_git_short()}"


def out_root_for_variant(out: str, weights: str = "imagenet",
                         stage1_only: bool = False) -> str:
    """Redirect the top-level `runs` dir to a variant-specific sibling so runs
    that differ in encoder weights or schedule depth don't overwrite each other.
    Only the leading `runs` component is renamed; the experiment subfolder is
    preserved. Suffixes compose (imagenet + 2-stage = no suffix):
        imagenet, 2-stage       -> runs/
        noisy-student, 2-stage  -> runs_noisystudent/
        imagenet, stage1-only   -> runs_stage1/
        noisy-student, stage1   -> runs_noisystudent_stage1/
    Idempotent, and a no-op when the leading component isn't `runs`.
    """
    suffix = ""
    if weights != "imagenet":
        suffix += "_" + weights.replace("-", "")
    if stage1_only:
        suffix += "_stage1"
    if not suffix:
        return out
    parts = Path(out).parts
    if parts and parts[0] == "runs":
        return str(Path("runs" + suffix, *parts[1:]))
    return out


def _git_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).parent, stderr=subprocess.DEVNULL).decode()
        return bool(out.strip())
    except Exception:
        return False


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "n/a"


# Description of each augmentation policy (see data.py:_augment). Kept here so it
# lands in run_meta.json; the git commit pins the exact code.
_AUG_COMMON = {
    "applies_to": "train split only (val/test deterministic)",
    "hflip_p": 0.5, "vflip_p": 0.5,
    "note": "geometry synced on image+mask; photometric on image only; "
            "letterbox + ImageNet normalization applied to all splits",
}
AUG_POLICIES = {
    "default": {**_AUG_COMMON, "rotation_deg": [-15, 15],
                "brightness_factor": [0.8, 1.2], "contrast_factor": [0.8, 1.2]},
    "strong": {**_AUG_COMMON, "rotation_deg": [-30, 30],
               "brightness_factor": [0.4, 1.6], "contrast_factor": [0.4, 1.6],
               "zoom_factor": [0.8, 1.2], "saturation_factor": [0.6, 1.4],
               "hue_shift_p": 0.3, "blur_p": 0.3},
}
LOSS_DESC = {
    "dicebce": "0.5*Dice + 0.5*BCEWithLogits",
    "focal_tversky": "Focal-Tversky (alpha=0.7, beta=0.3, gamma=1.333) -- "
                     "penalizes false-positive background (leakage) harder",
}

ARCH_DESC = {
    "unet": "Unet",
    "unetpp": "UnetPlusPlus (nested dense skip connections)",
}


def collect(run_id: str, args, model, splits: dict, device) -> dict:
    """Assemble the metadata dict written at the start of a run."""
    from model import ENCODER, count_params

    _, total = count_params(model)
    return {
        "run_id": run_id,
        "status": "running",
        "created": datetime.now().isoformat(timespec="seconds"),
        "command": "python " + " ".join(sys.argv),
        "seed": args.seed,
        "hyperparams": {
            "epochs1": args.epochs1, "epochs2": args.epochs2, "bs": args.bs,
            "lr1": args.lr1, "lr2": args.lr2, "patience": args.patience,
            "size": args.size, "amp": not args.no_amp,
            "max_batches": args.max_batches,
            "loss": args.loss, "aug": args.aug, "sched": args.sched,
            "arch": args.arch, "weights": getattr(args, "weights", "imagenet"),
            "stage1_only": getattr(args, "stage1_only", False),
        },
        "model": {
            "arch": ARCH_DESC.get(args.arch, args.arch) + " (segmentation-models-pytorch)",
            "encoder": ENCODER, "encoder_weights": getattr(args, "weights", "imagenet"),
            "classes": 1, "activation": "sigmoid",
            "params_total_M": round(total / 1e6, 3),
            "params_total": total,
        },
        "loss": LOSS_DESC.get(args.loss, args.loss),
        "schedule": (
            "stage1 only: encoder frozen (lr1, decoder only); "
            if getattr(args, "stage1_only", False)
            else "stage1: encoder frozen (lr1, decoder only); "
                 "stage2: full network unfrozen (lr2); "
        ) + "early stopping + best checkpoint on val iou_fg",
        "augmentations": {"mode": args.aug,
                          **AUG_POLICIES.get(args.aug, {})},
        "data": {
            "root": str(args.__dict__.get("root", "")) or "data.DATA_ROOT",
            "split_proportions": "70/15/15",
            "seed": args.seed,
            "n_train": len(splits["train"]),
            "n_val": len(splits["val"]),
            "n_test": len(splits["test"]),
        },
        "env": {
            "python": sys.version.split()[0],
            "torch": _pkg("torch"),
            "segmentation_models_pytorch": _pkg("segmentation-models-pytorch"),
            "timm": _pkg("timm"),
            "numpy": _pkg("numpy"),
            "device": str(device),
        },
        "git": {"commit": _git_short(), "dirty": _git_dirty()},
    }


def write(out: Path, meta: dict) -> None:
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))


def write_split_ids(out: Path, splits: dict) -> None:
    """Exact image ids per split -> a split is fully reproducible from this."""
    (out / "split_ids.json").write_text(json.dumps(
        {k: list(v) for k, v in splits.items()}, indent=2))


def finalize(out: Path, meta: dict, best: dict, history: list,
             duration_sec: float) -> None:
    """Update run_meta.json at the end with results + how long it took."""
    meta = dict(meta)
    meta["status"] = "done"
    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["duration_sec"] = round(duration_sec, 1)
    meta["epochs_run"] = len(history)
    meta["result_val_best"] = dict(best)  # best val metrics + stage/epoch
    write(out, meta)
