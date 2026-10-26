"""System identification: recover (mass, friction, damping, gain) from 'real' data."""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from scipy.optimize import differential_evolution

from .conditions import REAL_IDENTIFIABLE
from .envs import Physics, make_env
from .evaluate import save_json

warnings.filterwarnings("ignore")
H = 25


def collect(real: Physics, n_segments=20, seed=0):
    """Record (initial state, action sequence, observed response) from the real system."""
    rng = np.random.default_rng(seed)
    env = make_env(real, seed=seed)
    env.reset(seed=seed)
    u = env.unwrapped
    segs = []
    while len(segs) < n_segments:
        env.reset(seed=int(rng.integers(1e6)))
        for _ in range(int(rng.integers(0, 15))):                 # diversify the start state
            env.step(rng.uniform(-1, 1, 3))
        q, qd = u.data.qpos.copy(), u.data.qvel.copy()
        acts, obs_seq = rng.uniform(-1, 1, (H, 3)), []
        for a in acts:
            o, _, term, trunc, _ = env.step(a)
            obs_seq.append(o)
            if term or trunc:
                break
        if len(obs_seq) >= 8:
            segs.append(dict(q=q, qd=qd, acts=acts[:len(obs_seq)], obs=np.array(obs_seq)))
    return segs


def simulate(theta, segs):
    env = make_env(Physics.from_vec(theta))
    env.reset(seed=0)
    u, out = env.unwrapped, []
    for s in segs:
        env.reset(seed=0)
        u.set_state(s["q"], s["qd"])
        out.append(np.array([env.step(a)[0] for a in s["acts"]]))
    return out


def loss_fn(theta, segs, scale):
    sims = simulate(theta, segs)
    return float(np.mean([(((s["obs"] - sim) / scale) ** 2).mean() for s, sim in zip(segs, sims)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", type=int, default=20)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    real = REAL_IDENTIFIABLE                                      # hidden from the optimizer
    segs = collect(real, a.segments, a.seed)
    scale = np.concatenate([s["obs"] for s in segs]).std(0) + 1e-6
    bounds = [(0.5, 1.6), (0.4, 1.6), (0.5, 2.0), (0.5, 1.5)]
    res = differential_evolution(loss_fn, bounds, args=(segs, scale), maxiter=a.iters,
                                 popsize=10, seed=a.seed, polish=False, tol=1e-6)
    est, truth = res.x, real.vec()
    print("true     ", np.round(truth, 3))
    print("estimated", np.round(est, 3), " loss", round(res.fun, 5))
    print("loss at nominal:", round(loss_fn(np.ones(4), segs, scale), 5))
    save_json(dict(estimated=est.tolist(), truth=truth.tolist(), loss=res.fun),
              "results/sysid_params.json")


if __name__ == "__main__":
    main()
