"""Aggregate the per-seed threshold sweeps into leakage/iou_fg-vs-threshold
curves (mean +/- std over seeds), plus the parametric leakage-vs-iou_fg tradeoff.

Reads the threshold_sweep.json already saved per seed (no retrain / re-eval).
Usage: python plot_leakage_iou.py [--exp runs_noisystudent_stage1/effb0_unetpp_strong]
                                   [--split test|val|both]
"""
import argparse, glob, json, os
import numpy as np
import matplotlib.pyplot as plt

DEF_EXP = "runs_noisystudent_stage1/effb0_unetpp_strong"


def load_seeds(exp):
    """-> (thresholds, {split: {metric: array[seed, thr]}}), one row per seed."""
    files = sorted(glob.glob(os.path.join(exp, "*seed*", "threshold_sweep.json")))
    if not files:
        raise SystemExit(f"no threshold_sweep.json under {exp}")
    thr = None
    data = {}
    seeds = []
    for f in files:
        d = json.load(open(f))
        seeds.append(d["seed"])
        t = d["thresholds"]
        thr = t if thr is None else thr
        assert d["thresholds"] == thr, "threshold grids differ across seeds"
        for split, perthr in d["splits"].items():
            ds = data.setdefault(split, {})
            for ti, tk in enumerate([f"{x:.3f}" for x in thr]):
                for m, v in perthr[tk].items():
                    ds.setdefault(m, {}).setdefault(ti, []).append(v)
    # -> arrays [n_seeds, n_thr]
    out = {}
    for split, mets in data.items():
        out[split] = {m: np.array([[per[ti][s] for ti in range(len(thr))]
                                   for s in range(len(seeds))])
                      for m, per in mets.items()}
    return np.array(thr), out, seeds


def plot(exp, which):
    thr, data, seeds = load_seeds(exp)
    splits = ["val", "test"] if which == "both" else [which]
    print(f"{len(seeds)} seeds {seeds}, thresholds {thr.min()}..{thr.max()}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel 1: iou_fg & leakage vs threshold (mean +/- std band) ---
    ax = axes[0]
    colors = {"test": ("C0", "C3"), "val": ("C9", "C1")}
    for split in splits:
        ci, cl = colors[split]
        for met, c, ls in [("iou_fg", ci, "-"), ("leakage", cl, "--")]:
            a = data[split][met]
            mu, sd = a.mean(0), a.std(0)
            ax.plot(thr, mu, ls, color=c, label=f"{met} ({split})")
            ax.fill_between(thr, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("metric")
    ax.set_title(f"iou_fg / leakage vs threshold (mean +/- std, {len(seeds)} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- panel 2: parametric tradeoff leakage (x) vs iou_fg (y) ---
    ax = axes[1]
    for split in splits:
        ci, _ = colors[split]
        lk = data[split]["leakage"].mean(0)
        fg = data[split]["iou_fg"].mean(0)
        ax.plot(lk, fg, "-o", color=ci, ms=3, label=split)
        for t, x, y in zip(thr, lk, fg):
            if abs((t * 100) % 10) < 1e-6:  # annotate every 0.1
                ax.annotate(f"{t:.1f}", (x, y), fontsize=7,
                            textcoords="offset points", xytext=(4, 3))
    ax.set_xlabel("leakage (1 - precision)")
    ax.set_ylabel("iou_fg")
    ax.set_title("tradeoff: lower-left leakage vs iou_fg (labels = threshold)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(exp, "leakage_iou_curve.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default=DEF_EXP)
    p.add_argument("--split", default="both", choices=["test", "val", "both"])
    args = p.parse_args()
    plot(args.exp, args.split)
