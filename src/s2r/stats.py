"""Robust aggregate statistics (Agarwal et al., 2021 style) and the results table."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def iqm(x):
    """Interquartile mean: mean of the middle 50% of values."""
    x = np.sort(np.ravel(x))
    n = len(x)
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return float(x[lo:hi].mean())


def bootstrap_ci(x, stat=np.mean, n_boot=5000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    boots = [stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def load_results(folder="results"):
    out = {}
    for f in sorted(glob.glob(f"{folder}/*.json")):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict) and "real" in d:                 # method result files
            out[Path(f).stem] = d
    return out


def main():
    res = load_results()
    conds = ["nominal", "real"]
    print("| method | seeds | " + " | ".join(f"{c} (mean [95% CI])" for c in conds) + " |")
    print("|---|---|" + "---|" * len(conds))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (m, d) in enumerate(res.items()):
        cells = []
        for c in conds:
            v = np.array(d[c])
            lo, hi = bootstrap_ci(v)
            cells.append(f"{v.mean():.0f} [{lo:.0f}, {hi:.0f}]")
        print(f"| {m} | {len(d['real'])} | " + " | ".join(cells) + " |")
        v = np.array(d["real"])
        lo, hi = bootstrap_ci(v)
        ax.bar(i, v.mean(), yerr=[[v.mean() - lo], [hi - v.mean()]], capsize=4)
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels(list(res), rotation=30)
    ax.set_ylabel("return on 'real'")
    fig.tight_layout()
    fig.savefig("results/real_comparison.png", dpi=130)


if __name__ == "__main__":
    main()
