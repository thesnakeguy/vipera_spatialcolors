import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import runmeta
from data import make_datasets
from model import (build_model, set_encoder_trainable, freeze_frozen_batchnorm,
                   count_params, ENCODER)
from losses import get_loss, IoUMeter


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None,
              max_batches=0):
    train = optimizer is not None
    model.train(train)
    if train:
        # keep BN of any fully-frozen submodule (the encoder in stage 1) in eval
        # mode so its ImageNet running stats don't drift while it's "frozen".
        freeze_frozen_batchnorm(model)
    meter = IoUMeter()
    total_loss, n = 0.0, 0
    for bi, (x, y) in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast(
                device_type=device.type, enabled=scaler is not None):
            logits = model(x)
            loss = loss_fn(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        meter.update(logits.float(), y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(n, 1), meter.compute()


def train_stage(name, model, loaders, device, lr, epochs, patience,
                scaler, best, history, out, args, loss_fn):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(
                 opt, mode="max", factor=0.5, patience=4, min_lr=1e-7)
             if args.sched == "plateau" else None)
    tr_n, _ = count_params(model)
    print(f"\n=== {name} | lr={lr:g} | trainable={tr_n/1e6:.2f}M | "
          f"epochs<= {epochs} | patience={patience} ===")
    bad = 0
    stage_best = -1.0  # early-stop tracks improvement WITHIN this stage (see below)
    for ep in range(epochs):
        t = time.time()
        tr_loss, tr = run_epoch(model, loaders["train"], device, loss_fn, opt,
                                scaler, args.max_batches)
        va_loss, va = run_epoch(model, loaders["val"], device, loss_fn,
                                max_batches=args.max_batches)
        # Checkpoint on iou_fg (snake IoU) -- the metric we want to improve, not
        # the background-inflated iou_mean. best.pt = best across BOTH stages.
        saved = va["iou_fg"] > best["iou_fg"]
        if saved:
            best.update(iou_fg=va["iou_fg"], leakage=va["leakage"],
                        boundary_iou=va["boundary_iou"], iou_mean=va["iou_mean"],
                        stage=name, epoch=ep)
            torch.save({"model": model.state_dict(), "best": dict(best),
                        "encoder": ENCODER, "arch": args.arch, "weights": args.weights,
                        "size": args.size, "seed": args.seed,
                        "run_id": args.run_id}, out / "best.pt")
        # Early-stop on WITHIN-stage improvement, NOT the global best. Stage 2
        # unfreezes the encoder and dips ~0.06 below stage 1's peak before climbing
        # back; measuring patience against the global best kills it mid-climb,
        # before it can recover. Reset the counter on improvement over this stage's
        # own best instead, so a stage stops only when it genuinely plateaus.
        if va["iou_fg"] > stage_best:
            stage_best = va["iou_fg"]
            bad = 0
        else:
            bad += 1
        if sched is not None:
            sched.step(va["iou_fg"])
        history.append({"stage": name, "epoch": ep, "train_loss": tr_loss,
                        "val_loss": va_loss, **{f"val_{k}": v for k, v in va.items()}})
        print(f"[{name} {ep:3d}] {time.time()-t:5.0f}s  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
              f"val_iou_fg={va['iou_fg']:.4f}  leak={va['leakage']:.4f}  "
              f"bnd={va['boundary_iou']:.4f}{'  <- best' if saved else ''}")
        if patience and bad >= patience:
            print(f"  early stop ({patience} epochs no improvement)")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs1", type=int, default=100)
    p.add_argument("--epochs2", type=int, default=50)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr1", type=float, default=None,
                   help="stage-1 LR (default: 1e-3 with --sched plateau, else 2e-4)")
    p.add_argument("--lr2", type=float, default=2e-5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="cap batches/epoch (smoke test)")
    p.add_argument("--loss", type=str, default="dicebce",
                   choices=["dicebce", "focal_tversky"],
                   help="dicebce = baseline; focal_tversky = anti-leakage variant")
    p.add_argument("--aug", type=str, default="default",
                   choices=["default", "strong"],
                   help="train augmentation strength (strong adds zoom/hue/sat/blur)")
    p.add_argument("--sched", type=str, default="constant",
                   choices=["constant", "plateau"],
                   help="constant = baseline; plateau = ReduceLROnPlateau on val iou_fg")
    p.add_argument("--arch", type=str, default="unet",
                   choices=["unet", "unetpp"],
                   help="decoder: unet = baseline; unetpp = U-Net++ (nested skips)")
    p.add_argument("--weights", type=str, default="imagenet",
                   choices=["imagenet", "noisy-student"],
                   help="encoder pretrained weights; noisy-student lands in runs_noisystudent/")
    p.add_argument("--stage1-only", action="store_true",
                   help="train only stage 1 (frozen encoder, decoder); skip the "
                        "stage-2 unfreeze. Adds suffix _stage1 to the runs dir. "
                        "Stage 2 adds ~nothing once stage-1 BN is correct (see FINDINGS).")
    p.add_argument("--seed", type=int, default=42, help="data-split seed (also seeds torch)")
    p.add_argument("--out", type=str, default="runs/effb0_baseline",
                   help="experiment dir; each run lands in <out>/<run_id>")
    p.add_argument("--run-id", type=str, default=None,
                   help="unique run-folder name (default: auto timestamp-seed-githash)")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    # Variant runs land in a sibling top-level dir so they don't overwrite the
    # baseline: non-ImageNet weights add _<weights> and stage1-only adds _stage1
    # (e.g. runs/ -> runs_noisystudent_stage1/). Only the leading `runs` component
    # is renamed; the experiment subfolder is preserved.
    args.out = runmeta.out_root_for_variant(args.out, args.weights, args.stage1_only)

    # Stage-1 LR default tracks the scheduler: a plateau scheduler anneals the LR
    # automatically, so start high (1e-3); with a constant LR keep the safer 2e-4.
    # These are the bs=16 values (linearly scaled from the bs=8 base of 5e-4 /
    # 1e-4). An explicit --lr1 always wins.
    if args.lr1 is None:
        args.lr1 = 1e-3 if args.sched == "plateau" else 2e-4

    # Seed the global RNGs so weight init / augmentation are reproducible per seed.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # AMP only makes sense on CUDA; the GradScaler is CUDA-specific, so gate on
    # the device, not just --no-amp (a CPU fallback run would otherwise mismatch
    # the autocast device_type and break).
    use_amp = not args.no_amp and device.type == "cuda"
    run_id = args.run_id or runmeta.make_run_id(args.seed)
    args.run_id = run_id  # so train_stage (which only sees args) can stamp ckpts
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={device}  run_id={run_id}  out={out}  amp={use_amp}  "
          f"loss={args.loss}  aug={args.aug}")

    ds = make_datasets(size=args.size, seed=args.seed, aug=args.aug)
    loss_fn = get_loss(args.loss)
    loaders = {
        "train": DataLoader(ds["train"], batch_size=args.bs, shuffle=True,
                            num_workers=args.workers, pin_memory=True, drop_last=True),
        "val": DataLoader(ds["val"], batch_size=args.bs, shuffle=False,
                          num_workers=args.workers, pin_memory=True),
    }
    print(f"train={len(ds['train'])}  val={len(ds['val'])}  test={len(ds['test'])}")

    model = build_model(arch=args.arch, weights=args.weights).to(device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best = {"iou_fg": -1.0, "leakage": 1.0, "boundary_iou": -1.0,
            "iou_mean": -1.0, "stage": None, "epoch": -1}
    history = []

    # Provenance: write run_meta.json + split_ids.json up front so even a
    # crashed run is traceable; finalized with results at the end.
    splits = {k: ds[k].ids for k in ("train", "val", "test")}
    meta = runmeta.collect(run_id, args, model, splits, device)
    runmeta.write(out, meta)
    runmeta.write_split_ids(out, splits)
    t0 = time.time()

    # Stage 1: encoder frozen, decoder only
    set_encoder_trainable(model, False)
    train_stage("stage1", model, loaders, device, args.lr1, args.epochs1,
                args.patience, scaler, best, history, out, args, loss_fn)

    # Stage 2: unfreeze whole network, lower LR. Skippable -- once stage-1 BN is
    # handled correctly, unfreezing the encoder only recovers the unfreeze dip back
    # to ~par and adds no net gain on this small dataset (see FINDINGS).
    if not args.stage1_only:
        set_encoder_trainable(model, True)
        train_stage("stage2", model, loaders, device, args.lr2, args.epochs2,
                    args.patience, scaler, best, history, out, args, loss_fn)

    json.dump(history, open(out / "history.json", "w"), indent=2)
    runmeta.finalize(out, meta, best, history, time.time() - t0)
    print(f"\nBEST val_iou_fg={best['iou_fg']:.4f}  leakage={best['leakage']:.4f}  "
          f"boundary_iou={best['boundary_iou']:.4f}  (iou_mean={best['iou_mean']:.4f})  "
          f"@ {best['stage']} epoch {best['epoch']}")
    print(f"saved: {out/'best.pt'}  |  run_meta.json written  |  "
          f"improve iou_fg + lower leakage (orig iou_mean ~0.893)")


if __name__ == "__main__":
    main()
