"""Run the full pipeline across several data-split seeds and aggregate.

Robustness over a single lucky split: for each seed it trains (train.py) then
evaluates the held-out test set (eval.py), and finally aggregates the per-seed
test metrics into a mean +/- std summary. A result reported as a distribution
over splits is far more trustworthy than one number from random_state=42.

To keep disk sane, per-image prediction subplots are saved ONLY for the
original project seed (42); every seed still gets metrics + training curves.

  python run_seeds.py                        # seeds 42 0 1 2 3, full schedule
  python run_seeds.py --skip-train           # re-aggregate from existing ckpts
  python run_seeds.py --seeds 42 0 --epochs1 1 --epochs2 1 --max-batches 3  # smoke
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

import runmeta

HERE = Path(__file__).parent
KEYS = ["iou_fg", "leakage", "boundary_iou", "recall", "iou_mean"]
SWEEP_METRICS = ["iou_fg", "leakage", "recall", "boundary_iou"]
FIG_SEED = 42  # original project seed -> the only one we save subplots for


def default_grid() -> list[float]:
    """Threshold grid: 0.30 .. 0.90 step 0.05. Mirrors eval.default_grid (kept
    in sync by formula so run_seeds needn't import torch via eval)."""
    return [round(0.30 + 0.05 * i, 2) for i in range(13)]


def run(cmd):
    cmd = [str(c) for c in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def latest_run_for_seed(base: Path, seed: int) -> Path:
    """Most recent run folder for a seed (run-ids are timestamp-sorted)."""
    runs = sorted(base.glob(f"*-seed{seed}-*"))
    if not runs:
        raise FileNotFoundError(f"no run folder for seed {seed} under {base}")
    return runs[-1]


def calibrate(val: dict, grid: list[float], floor: float) -> float:
    """Pick the operating threshold: minimum leakage among thresholds whose VAL
    recall is >= floor. If the floor is unreachable, fall back to the
    highest-recall threshold (the best we can do)."""
    feasible = [t for t in grid if val[f"{t:.3f}"]["recall"] >= floor]
    if feasible:
        return min(feasible, key=lambda t: val[f"{t:.3f}"]["leakage"])
    return max(grid, key=lambda t: val[f"{t:.3f}"]["recall"])


def run_sweep(args, base: Path, py: str) -> None:
    """Re-eval existing checkpoints across a threshold grid (no retraining),
    calibrate on val (min leakage s.t. recall >= floor) and report on test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = args.thresholds or default_grid()
    gkeys = [f"{t:.3f}" for t in grid]
    floor = args.recall_floor

    sweeps, run_ids = {}, {}
    for seed in args.seeds:
        run_dir = latest_run_for_seed(base, seed)
        run_ids[seed] = run_dir.name
        # --sweep-out run_dir keeps the per-seed sweep next to the checkpoint
        # (run_seeds layout); standalone `eval.py --sweep` defaults to an eval/ dir.
        cmd = [py, "eval.py", "--ckpt", run_dir / "best.pt", "--sweep",
               "--sweep-out", run_dir,
               "--sweep-splits", "val", "test",
               "--thresholds", *[str(t) for t in grid]]
        if args.max_batches:
            cmd += ["--max-batches", args.max_batches]
        run(cmd)
        sweeps[seed] = json.loads(
            (run_dir / "threshold_sweep.json").read_text())["splits"]

    def agg(per_seed):
        out = {}
        for k in SWEEP_METRICS:
            vals = [per_seed[s][k] for s in args.seeds]
            out[k] = {"mean": statistics.fmean(vals),
                      "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}
        return out

    # mean TEST curve across seeds (for the tradeoff plot / table)
    curve = {k: [statistics.fmean(sweeps[s]["test"][gk][k] for s in args.seeds)
                 for gk in gkeys] for k in SWEEP_METRICS}

    # per-seed calibration at the chosen floor; recommend a single deployable
    # threshold = median of the per-seed optimum (snapped to the grid).
    tstar = {s: calibrate(sweeps[s]["val"], grid, floor) for s in args.seeds}
    test_at_tstar = {s: sweeps[s]["test"][f"{tstar[s]:.3f}"] for s in args.seeds}
    med = statistics.median(list(tstar.values()))
    global_t = min(grid, key=lambda t: abs(t - med))
    test_at_global = {s: sweeps[s]["test"][f"{global_t:.3f}"] for s in args.seeds}

    # how the operating point + test metrics move with the recall floor
    floors = sorted({0.85, 0.88, 0.90, 0.92, floor})
    sens = {}
    for fl in floors:
        ts = {s: calibrate(sweeps[s]["val"], grid, fl) for s in args.seeds}
        ta = {s: sweeps[s]["test"][f"{ts[s]:.3f}"] for s in args.seeds}
        sens[fl] = {"median_t": statistics.median(list(ts.values())),
                    "test_agg": agg(ta)}

    summary = {
        "seeds": args.seeds, "run_ids": run_ids, "thresholds": grid,
        "recall_floor": floor,
        "mean_test_curve": {"thresholds": grid, **curve},
        "per_seed_tstar": {str(s): tstar[s] for s in args.seeds},
        "test_at_tstar_agg": agg(test_at_tstar),
        "recommended_threshold": global_t,
        "test_at_recommended_agg": agg(test_at_global),
        "baseline_thr_0.5": agg({s: sweeps[s]["test"]["0.500"] for s in args.seeds})
        if "0.500" in sweeps[args.seeds[0]]["test"] else None,
        "floor_sensitivity": {f"{fl:.2f}": sens[fl] for fl in floors},
    }
    (base / "threshold_summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, c in [("iou_fg", "C0"), ("leakage", "C3"), ("recall", "C2"),
                 ("boundary_iou", "C1")]:
        ax.plot(grid, curve[k], marker="o", label=k, color=c)
    ax.axvline(global_t, ls="--", color="gray", lw=1,
               label=f"recommended {global_t:.2f}")
    ax.axhline(floor, ls=":", color="C2", lw=1, label=f"recall floor {floor:.2f}")
    ax.set_xlabel("decision threshold"); ax.set_ylabel("mean TEST metric")
    ax.set_title(f"threshold sweep (best setup, {len(args.seeds)} seeds)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(base / "threshold_curve.png", dpi=120)
    plt.close(fig)

    print("\n=== Mean TEST metrics vs threshold (across seeds) ===")
    print(f"  {'thr':>5} {'iou_fg':>8} {'leakage':>8} {'recall':>8} {'bnd':>8}")
    for i, t in enumerate(grid):
        print(f"  {t:>5.2f} {curve['iou_fg'][i]:>8.4f} {curve['leakage'][i]:>8.4f} "
              f"{curve['recall'][i]:>8.4f} {curve['boundary_iou'][i]:>8.4f}")
    a = summary["test_at_tstar_agg"]
    print(f"\n=== Calibrated: min leakage s.t. VAL recall >= {floor} "
          f"(per-seed val -> test) ===")
    print(f"  per-seed t* = {[round(tstar[s], 2) for s in args.seeds]}")
    print(f"  test @ t*  : iou_fg {a['iou_fg']['mean']:.4f}+/-{a['iou_fg']['std']:.4f}"
          f"  leakage {a['leakage']['mean']:.4f}+/-{a['leakage']['std']:.4f}"
          f"  recall {a['recall']['mean']:.4f}")
    b = summary["test_at_recommended_agg"]
    print(f"  recommended single threshold = {global_t:.2f} (median of per-seed t*)")
    print(f"  test @ {global_t:.2f} : iou_fg {b['iou_fg']['mean']:.4f}  "
          f"leakage {b['leakage']['mean']:.4f}  recall {b['recall']['mean']:.4f}")
    print(f"\n=== Floor sensitivity (per-seed calibrated test aggregate) ===")
    print(f"  {'floor':>6} {'med_t':>6} {'iou_fg':>8} {'leakage':>8} {'recall':>8}")
    for fl in floors:
        ta = sens[fl]["test_agg"]
        mark = " <- default" if abs(fl - floor) < 1e-9 else ""
        print(f"  {fl:>6.2f} {sens[fl]['median_t']:>6.2f} {ta['iou_fg']['mean']:>8.4f} "
              f"{ta['leakage']['mean']:>8.4f} {ta['recall']['mean']:>8.4f}{mark}")
    print(f"\nwrote {base / 'threshold_summary.json'}")
    print(f"wrote {base / 'threshold_curve.png'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    p.add_argument("--out", type=str, default="runs/effb0_baseline")
    p.add_argument("--epochs1", type=int, default=100)
    p.add_argument("--epochs2", type=int, default=50)
    p.add_argument("--max-batches", type=int, default=0,
                   help="cap batches/epoch (smoke test)")
    # forwarded to train.py so a variant is fully expressible from the runner
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lr1", type=float, default=None,
                   help="stage-1 LR (default: 1e-3 with --sched plateau, else 2e-4)")
    p.add_argument("--lr2", type=float, default=2e-5)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--loss", type=str, default="dicebce",
                   choices=["dicebce", "focal_tversky"])
    p.add_argument("--aug", type=str, default="default",
                   choices=["default", "strong"])
    p.add_argument("--sched", type=str, default="constant",
                   choices=["constant", "plateau"])
    p.add_argument("--arch", type=str, default="unet",
                   choices=["unet", "unetpp"])
    p.add_argument("--weights", type=str, default="imagenet",
                   choices=["imagenet", "noisy-student"],
                   help="encoder pretrained weights; noisy-student lands in runs_noisystudent/")
    p.add_argument("--stage1-only", action="store_true",
                   help="train only stage 1 (skip stage-2 unfreeze); adds suffix _stage1")
    p.add_argument("--skip-train", action="store_true",
                   help="evaluate + aggregate existing checkpoints only")
    p.add_argument("--sweep", action="store_true",
                   help="threshold sweep: re-eval existing checkpoints across a "
                        "grid (no retraining), calibrate on val, report on test")
    p.add_argument("--recall-floor", type=float, default=0.88,
                   help="--sweep: min VAL recall to keep when minimizing leakage")
    p.add_argument("--thresholds", type=float, nargs="+", default=None,
                   help="--sweep: threshold grid (default 0.30..0.90 step 0.05)")
    args = p.parse_args()
    # Stage-1 LR default tracks the scheduler (see train.py): 1e-3 with plateau,
    # else 2e-4 (bs=16 values); an explicit --lr1 overrides. Resolved here so the
    # value forwarded to train.py and recorded in the summary is consistent.
    if args.lr1 is None:
        args.lr1 = 1e-3 if args.sched == "plateau" else 2e-4

    py = sys.executable
    # Mirror train.py: variant runs aggregate under a suffixed sibling dir (e.g.
    # runs_noisystudent_stage1/) so eval/glob/summary all target the same dir as
    # the per-seed runs.
    args.out = runmeta.out_root_for_variant(args.out, args.weights, args.stage1_only)
    base = HERE / args.out
    if args.sweep:
        run_sweep(args, base, py)
        return
    per_seed, run_ids = {}, {}
    for seed in args.seeds:
        if args.skip_train:
            run_dir = latest_run_for_seed(base, seed)
        else:
            run_id = runmeta.make_run_id(seed)
            run_dir = base / run_id
            cmd = [py, "train.py", "--seed", seed, "--run-id", run_id,
                   "--out", args.out, "--epochs1", args.epochs1,
                   "--epochs2", args.epochs2, "--max-batches", args.max_batches,
                   "--bs", args.bs, "--lr1", args.lr1, "--lr2", args.lr2,
                   "--patience", args.patience, "--loss", args.loss,
                   "--aug", args.aug, "--sched", args.sched,
                   "--arch", args.arch, "--weights", args.weights]
            if args.stage1_only:
                cmd.append("--stage1-only")
            run(cmd)
        run_ids[seed] = run_dir.name
        eval_cmd = [py, "eval.py", "--ckpt", run_dir / "best.pt"]
        if seed != FIG_SEED:
            eval_cmd.append("--no-figs")
        run(eval_cmd)
        per_seed[seed] = json.loads(
            (run_dir / "test_metrics.json").read_text())["metrics"]

    summary = {"seeds": args.seeds, "run_ids": run_ids,
               "per_seed": per_seed, "aggregate": {}}
    for k in KEYS:
        vals = [per_seed[s][k] for s in args.seeds]
        summary["aggregate"][k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals),
        }
    (base / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== TEST-set metrics across seeds (mean +/- std) ===")
    for k in KEYS:
        a = summary["aggregate"][k]
        print(f"  {k:13s} {a['mean']:.4f} +/- {a['std']:.4f}  "
              f"[min {a['min']:.4f}, max {a['max']:.4f}]")
    print(f"\nwrote {base / 'summary.json'}")
    if FIG_SEED in run_ids:
        print(f"subplots for the original seed {FIG_SEED}: "
              f"{base / run_ids[FIG_SEED] / 'preds'}/")


if __name__ == "__main__":
    main()
