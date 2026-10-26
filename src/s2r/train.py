"""Train PPO on Hopper, optionally with domain randomization / privileged obs."""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from .envs import Physics, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--dr", type=float, default=0.0, help="randomization scale in [0,1]")
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--sim-physics", type=float, nargs=4, default=None,
                    metavar=("MASS", "FRIC", "DAMP", "GAIN"),
                    help="train in a fixed identified simulator instead of nominal")
    ap.add_argument("--out", default="models")
    a = ap.parse_args()

    phys = Physics.from_vec(a.sim_physics) if a.sim_physics else Physics()
    env = make_vec_env(lambda: make_env(phys, a.dr or None, a.privileged, a.seed),
                       n_envs=4, seed=a.seed)
    model = PPO("MlpPolicy", env, seed=a.seed, device="cpu", verbose=1,
                n_steps=1024, batch_size=256, learning_rate=3e-4)
    model.learn(a.steps)
    Path(a.out).mkdir(exist_ok=True)
    model.save(f"{a.out}/{a.name}_s{a.seed}")


if __name__ == "__main__":
    main()
