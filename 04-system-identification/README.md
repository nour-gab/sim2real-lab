# 04 · System identification and real2sim

**Goal:** recover the hidden physical parameters from a few "real" trajectories, then retrain in the *corrected* simulator.

## Run it
```bash
export PYTHONPATH=src
python -m s2r.sysid                      # -> results/sysid_params.json
SIM=$(python -c "import json;print(*json.load(open('results/sysid_params.json'))['estimated'])")
for s in 0 1 2; do python -m s2r.train --name sysid --seed $s --steps 300000 --sim-physics $SIM; done
python -m s2r.eval_suite --method sysid
```
The optimizer never sees the true parameters. It only sees recorded start states, action sequences and observed responses.

## The math

**Data.** Segments $i=1..N$ with start state $x_{i,0}$, actions $a_{i,0:H-1}$, and observed responses $o^{real}_{i,1:H}$.

**Estimator (nonlinear least squares).**
$$\hat\theta=\arg\min_{\theta\in\Theta}\ \frac1{N}\sum_{i=1}^{N}\sum_{t=1}^{H}\Big\|W^{-1}\big(o^{real}_{i,t}-o^{sim}_{i,t}(\theta)\big)\Big\|^2,$$
where $W$ is a per-dimension scale (we use the standard deviation of the real observations) so velocities do not dominate positions. Short horizons $H$ avoid chaotic divergence from contacts.

**Why a global optimizer.** Contact makes $o^{sim}(\theta)$ non-smooth, so gradients are unreliable and local optimizers get stuck. We use differential evolution (derivative-free, population-based). With a differentiable simulator you could use gradients.

**Identifiability.** Linearize the response, $o(\theta+\delta)\approx o(\theta)+J\delta$ with sensitivity matrix $J=\partial o/\partial\theta$. With Gaussian noise $\sigma^2$ the Fisher information is
$$F=\tfrac1{\sigma^2}J^\top J,\qquad \mathrm{Cov}(\hat\theta)\succeq F^{-1}\ \ \text{(Cramér-Rao)}.$$
Parameters that change the response weakly, or in the same way as another parameter (for example mass and actuator gain both change acceleration), give a small or near-singular $F$ and large uncertainty. Excitation matters: *choose actions that make $J$ informative.* Random actions are a baseline, not an optimum.

**Limits.** In the smoke test with a tiny budget the optimizer recovered mass, friction and gain well but damping poorly: a live demonstration of weak identifiability. Delay and noise are not identified here. Exercise: estimate delay by cross-correlating commanded and observed motion.

**Real2sim and virtual twins.** This is the loop a virtual twin needs: measure, calibrate the model, re-validate. The twin supplies geometry and nominal parameters; data corrects them.

## Exercises
1. Increase `--segments` and `--iters`. Plot estimation error against the amount of data.
2. Design actions that excite damping more. Does its estimate improve?
3. Combine this with section 03: randomize in a narrow band around $\hat\theta$.

## Article seed
*"System identification for sim2real: what you can and cannot recover from trajectories."*

**References:** Ljung, *System Identification: Theory for the User*; Hwangbo et al. (2019), actuator network; Peng et al. (2018).
