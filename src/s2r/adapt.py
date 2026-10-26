"""RMA-lite: privileged teacher -> history-based adapter -> student.

Phase 1 (train.py --dr 1.0 --privileged): teacher sees theta in its observation.
Phase 2 (this file): learn phi(history of obs, actions) ~ theta, then run the
teacher with phi's estimate in place of the true theta.
"""
from __future__ import annotations

import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from .conditions import TARGETS
from .envs import CONT, make_env
from .evaluate import save_json

OBS, ACT, K = 11, 3, 8            # core obs dim, action dim, history length
NC = len(CONT)


class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K * (OBS + ACT), 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, NC))

    def forward(self, x):
        return self.net(x)


class History:
    def __init__(self):
        self.h = deque([np.zeros(OBS + ACT, np.float32)] * K, maxlen=K)

    def push(self, obs_core, act):
        self.h.append(np.concatenate([obs_core, act]).astype(np.float32))

    def vec(self):
        return np.concatenate(list(self.h))


def collect(teacher, episodes=60, seed=0):
    """Roll out the teacher under randomization; record (history, true theta-1)."""
    env = make_env(dr_scale=1.0, privileged=True, seed=seed)
    X, Y = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        hist, done = History(), False
        while not done:
            a, _ = teacher.predict(obs, deterministic=False)
            X.append(hist.vec())
            Y.append(obs[OBS:])                                 # privileged label
            obs, _, term, trunc, _ = env.step(a)
            hist.push(obs[:OBS], a)
            done = term or trunc
    return np.array(X), np.array(Y, dtype=np.float32)


def fit(X, Y, epochs=40, seed=0):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, Yt = torch.tensor((X - mu) / sd, dtype=torch.float32), torch.tensor(Y)
    n = len(Xt)
    idx = torch.randperm(n)
    tr, va = idx[: int(0.9 * n)], idx[int(0.9 * n):]
    net, opt = Adapter(), None
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for e in range(epochs):
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        if e % 10 == 9:
            with torch.no_grad():
                print(f"epoch {e+1}: val MSE {((net(Xt[va]) - Yt[va]) ** 2).mean():.4f}")
    return net, mu, sd


def eval_student(teacher, net, mu, sd, physics, episodes=10, seed=1000, oracle=False):
    env = make_env(physics, privileged=True, seed=seed)
    rets = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        hist, done, R = History(), False, 0.0
        while not done:
            o = obs.copy()
            if not oracle:                                      # swap true theta for phi(history)
                with torch.no_grad():
                    z = torch.tensor(((hist.vec() - mu) / sd)[None], dtype=torch.float32)
                    o[OBS:] = net(z).numpy()[0]
            a, _ = teacher.predict(o, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            hist.push(obs[:OBS], a)
            R += r
            done = term or trunc
        rets.append(R)
    return float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--eval-episodes", type=int, default=10)
    a = ap.parse_args()
    teacher = PPO.load(a.teacher, device="cpu")
    X, Y = collect(teacher, a.episodes)
    print("dataset", X.shape, Y.shape)
    net, mu, sd = fit(X, Y)
    out = {"teacher_oracle": {}, "student": {}}
    for c, p in TARGETS.items():
        out["teacher_oracle"][c] = [eval_student(teacher, net, mu, sd, p, a.eval_episodes, oracle=True)]
        out["student"][c] = [eval_student(teacher, net, mu, sd, p, a.eval_episodes)]
        print(c, "oracle", round(out["teacher_oracle"][c][0]), "student", round(out["student"][c][0]))
    for k, v in out.items():
        save_json(v, f"results/{k}.json")


if __name__ == "__main__":
    main()
