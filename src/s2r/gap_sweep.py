"""Sensitivity sweep: how fast does performance fall as ONE parameter drifts?"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from .envs import Physics
from .evaluate import eval_model, save_json

SWEEPS = {
    "mass":      [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
    "friction":  [0.4, 0.6, 0.8, 1.0, 1.2, 1.5],
    "damping":   [0.5, 0.75, 1.0, 1.5, 2.0],
    "gain":      [0.6, 0.75, 0.9, 1.0, 1.1, 1.3],
    "delay":     [0, 1, 2, 3, 4],
    "obs_noise": [0.0, 0.01, 0.02, 0.05, 0.1],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--out", default="results/sweep")
    a = ap.parse_args()
    model = PPO.load(a.model, device="cpu")
    base = eval_model(model, Physics(), a.episodes).mean()
    res = {"nominal_return": float(base)}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (name, vals) in zip(axes.ravel(), SWEEPS.items()):
        rets = [float(eval_model(model, Physics(**{name: v}), a.episodes).mean()) for v in vals]
        res[name] = dict(values=vals, returns=rets, gap=[(base - r) / abs(base) for r in rets])
        ax.plot(vals, rets, "o-")
        ax.axhline(base, ls="--", c="gray")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(f"{a.out}.png", dpi=130)
    save_json(res, f"{a.out}.json")
    print(f"saved {a.out}.png / .json  (nominal return {base:.0f})")


if __name__ == "__main__":
    main()
