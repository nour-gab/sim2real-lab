"""Hopper with controllable physics: the 'source' sim and a hidden 'real' proxy.

theta = (mass, friction, damping, gain, delay, obs_noise)
Continuous parts scale MuJoCo model fields; delay and noise act on the I/O.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

ENV_ID = "Hopper-v5"
CONT = ["mass", "friction", "damping", "gain"]          # identifiable, continuous
BASE_RANGES = {"mass": (0.5, 1.5), "friction": (0.5, 1.5),
               "damping": (0.5, 1.5), "gain": (0.7, 1.3)}


@dataclass
class Physics:
    mass: float = 1.0
    friction: float = 1.0
    damping: float = 1.0
    gain: float = 1.0
    delay: int = 0
    obs_noise: float = 0.0

    def vec(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in CONT], dtype=np.float32)

    @staticmethod
    def from_vec(v, **kw) -> "Physics":
        return Physics(**{k: float(x) for k, x in zip(CONT, v)}, **kw)


class PhysicsWrapper(gym.Wrapper):
    """Applies a Physics setting; optionally appends (theta-1) to the observation."""

    def __init__(self, env, physics: Physics | None = None, privileged=False, seed=0):
        super().__init__(env)
        m = env.unwrapped.model
        self._m0, self._i0 = m.body_mass.copy(), m.body_inertia.copy()
        self._f0, self._d0 = m.geom_friction.copy(), m.dof_damping.copy()
        self.privileged = privileged
        self.rng = np.random.default_rng(seed)
        self._act_dim = env.action_space.shape[0]
        if privileged:
            lo = np.concatenate([env.observation_space.low, -np.ones(len(CONT))]).astype(np.float32)
            hi = np.concatenate([env.observation_space.high, 3 * np.ones(len(CONT))]).astype(np.float32)
            self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)
        self.set_physics(physics or Physics())

    def set_physics(self, p: Physics):
        m = self.env.unwrapped.model
        m.body_mass[:] = self._m0 * p.mass
        m.body_inertia[:] = self._i0 * p.mass          # inertia scales with mass
        m.geom_friction[:] = self._f0 * p.friction
        m.dof_damping[:] = self._d0 * p.damping
        self.p = p

    def _obs(self, obs):
        if self.p.obs_noise > 0:
            obs = obs + self.rng.normal(0, self.p.obs_noise, obs.shape)
        if self.privileged:
            obs = np.concatenate([obs, self.p.vec() - 1.0])
        return obs.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self._buf = deque([np.zeros(self._act_dim, np.float32)] * self.p.delay)
        obs, info = self.env.reset(seed=seed, options=options)
        return self._obs(obs), info

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32) * self.p.gain, -1.0, 1.0)
        if self.p.delay > 0:
            self._buf.append(a)
            a = self._buf.popleft()
        obs, r, term, trunc, info = self.env.step(a)
        return self._obs(obs), r, term, trunc, info


class DRWrapper(PhysicsWrapper):
    """Domain randomization: resample theta at every reset. scale in [0, 1]."""

    def __init__(self, env, scale=1.0, privileged=False, seed=0):
        self.scale = scale
        super().__init__(env, Physics(), privileged, seed)

    def sample(self) -> Physics:
        s, kw = self.scale, {}
        for k, (lo, hi) in BASE_RANGES.items():
            kw[k] = self.rng.uniform(1 - (1 - lo) * s, 1 + (hi - 1) * s)
        kw["delay"] = int(self.rng.integers(0, round(3 * s) + 1))
        kw["obs_noise"] = float(self.rng.uniform(0, 0.02 * s))
        return Physics(**kw)

    def reset(self, *, seed=None, options=None):
        self.set_physics(self.sample())
        return super().reset(seed=seed, options=options)


def make_env(physics: Physics | None = None, dr_scale: float | None = None,
             privileged=False, seed=0):
    env = gym.make(ENV_ID)
    if dr_scale is not None and dr_scale > 0:
        return DRWrapper(env, dr_scale, privileged, seed)
    return PhysicsWrapper(env, physics, privileged, seed)
