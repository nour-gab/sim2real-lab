"""Evaluate every seed of a method on every target condition."""
from __future__ import annotations

import argparse
import glob

from stable_baselines3 import PPO

from .conditions import TARGETS
from .evaluate import eval_model, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="model file prefix, e.g. base, dr05")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--models", default="models")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.models}/{a.method}_s*.zip"))
    assert files, f"no models match {a.models}/{a.method}_s*.zip"
    out = {c: [] for c in TARGETS}
    for f in files:
        m = PPO.load(f, device="cpu")
        for c, p in TARGETS.items():
            out[c].append(float(eval_model(m, p, a.episodes).mean()))   # one number per seed
        print(f, {c: round(v[-1]) for c, v in out.items()})
    save_json(out, f"results/{a.method}.json")


if __name__ == "__main__":
    main()
