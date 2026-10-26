# 01 · Foundations: MDPs, policy gradients and PPO

**Goal:** train a PPO policy on Hopper in the nominal simulator and understand every term in the objective you are optimizing.

## Run it
```bash
export PYTHONPATH=src
python -m s2r.train --name base --seed 0 --steps 300000
# watch ep_rew_mean in the log; it should climb from ~20 toward the high hundreds or more
```
`src/s2r/train.py` wraps Stable-Baselines3 PPO with 4 parallel environments.

## The math

**Markov decision process.** A tuple $(\mathcal S,\mathcal A,P,r,\gamma)$ with transition kernel $P(s'\mid s,a)$, reward $r(s,a)$ and discount $\gamma\in[0,1)$. A policy $\pi_\phi(a\mid s)$ induces trajectories $\tau=(s_0,a_0,s_1,\dots)$.

**Objective.**
$$J(\phi)=\mathbb E_{\tau\sim\pi_\phi}\Big[\sum_{t\ge 0}\gamma^t r(s_t,a_t)\Big]$$

**Value functions.** $V^\pi(s)=\mathbb E_\pi[\sum_k\gamma^k r_{t+k}\mid s_t=s]$, $Q^\pi(s,a)=r(s,a)+\gamma\,\mathbb E[V^\pi(s')]$, advantage $A^\pi=Q^\pi-V^\pi$. Bellman equation: $V^\pi(s)=\mathbb E_{a\sim\pi}\big[r(s,a)+\gamma\mathbb E_{s'}V^\pi(s')\big]$.

**Policy gradient theorem.**
$$\nabla_\phi J=\mathbb E_{\tau}\Big[\sum_t \nabla_\phi\log\pi_\phi(a_t\mid s_t)\,\hat A_t\Big]$$
Sketch of why: $\nabla_\phi \mathbb E_\tau[R(\tau)]=\mathbb E_\tau[R(\tau)\nabla_\phi\log p_\phi(\tau)]$ (the log-derivative trick) and the dynamics terms in $\log p_\phi(\tau)$ do not depend on $\phi$. Subtracting a baseline $b(s_t)$ leaves the gradient **unbiased** because $\mathbb E_a[\nabla\log\pi(a\mid s)]=0$, while reducing variance.

**Generalized advantage estimation.** With TD residual $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$,
$$\hat A_t^{GAE(\gamma,\lambda)}=\sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l}.$$
$\lambda=0$ is low variance and high bias; $\lambda=1$ is Monte Carlo (unbiased, high variance).

**PPO clipped objective** (Schulman et al., 2017). With ratio $\rho_t(\phi)=\frac{\pi_\phi(a_t\mid s_t)}{\pi_{\phi_{old}}(a_t\mid s_t)}$:
$$L^{CLIP}(\phi)=\mathbb E_t\Big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big]$$
The clip removes the incentive to move the policy far from the one that collected the data, a cheap approximation of a trust region.

## Why PPO fits robotics simulation
It is on-policy, simple and stable, and its throughput scales with the number of parallel environments (Rudin et al., 2021). SAC is more sample-efficient and off-policy, which matters when each sample is a real robot step.

## Exercises
1. Show $\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)]=0$ and conclude the baseline is unbiased.
2. Train with `--steps 100000` and `1000000`. Plot the learning curves. What changes?
3. Implement REINFORCE with a baseline in 60 lines and compare with PPO.

## Article seed
*"From the policy gradient theorem to PPO's clip: what each term does and what breaks without it."* Include the derivation and one learning-curve plot.

**References:** Sutton and Barto, *Reinforcement Learning* (2018); Schulman et al., *Proximal Policy Optimization Algorithms* (2017); Schulman et al., *GAE* (2015).
