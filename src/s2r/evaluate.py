"""Rollouts, metrics and the sim2real gap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .envs import Physics, make_env


def rollout(policy_fn, env, episodes=10, seed=1000):
    """Return an array of episode returns."""
    out = []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done, R = False, 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(policy_fn(obs))
            R += r
            done = term or trunc
        out.append(R)
    return np.array(out)


def sb3_policy(model):
    return lambda o: model.predict(o, deterministic=True)[0]


def eval_model(model, physics: Physics, episodes=10, seed=1000, privileged=False):
    env = make_env(physics, privileged=privileged, seed=seed)
    return rollout(sb3_policy(model), env, episodes, seed)


def gap(source_return: float, target_return: float) -> float:
    """Normalized gap: 0 = no loss, 1 = total loss."""
    return float((source_return - target_return) / max(abs(source_return), 1e-8))


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))
