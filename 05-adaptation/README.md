# 05 · Adaptation: privileged teacher and history-based student

**Goal:** make the policy *infer* the hidden physics from recent experience, instead of averaging over it. This is a minimal version of RMA (Kumar et al., 2021).

## Run it
```bash
export PYTHONPATH=src
python -m s2r.train --name teacher --seed 0 --steps 300000 --dr 1.0 --privileged
python -m s2r.adapt --teacher models/teacher_s0
# -> results/teacher_oracle.json (true theta given) and results/student.json (theta inferred)
```

## The math

**Phase 1: teacher.** Train $\pi(a\mid o,z)$ with a *privileged* latent $z=\theta-1\in\mathbb R^4$ (available in simulation only), under domain randomization. Knowing $z$ the policy can specialize: it is no longer a compromise across $\theta$.

**Phase 2: student.** Hidden parameters leave a trace in the dynamics. Since
$$o_{t+1}=f(o_t,a_t;\theta),$$
the recent history $h_t=(o_{t-K+1:t},a_{t-K+1:t})$ carries information about $\theta$. Learn an adapter $\phi$ by supervised regression:
$$\min_\phi\ \mathbb E\big\|\phi(h_t)-z\big\|^2,$$
then deploy $\pi(a\mid o,\hat z_t)$ with $\hat z_t=\phi(h_t)$. Data comes from teacher rollouts under randomization.

**Why it can work.** If $\theta\mapsto$ (response to actions) is injective, $\theta$ is a function of the history in the noise-free limit, so a regressor can in principle recover it (compare identifiability in section 04). Under noise, the achievable error is bounded below by the Bayes-optimal estimator's variance.

**Error propagation (informal).** If the teacher's return is $L$-Lipschitz in $z$, then
$$|J(\pi_z)-J(\pi_{\hat z})|\lesssim L\ \mathbb E\|z-\hat z\|,$$
so the adapter's validation error translates into lost return. The teacher with the true $z$ (`teacher_oracle`) is the upper bound for the student.

**Distribution mismatch.** The adapter is trained on states visited by the *teacher with true $z$*, but at deployment the student drives. A DAgger-style fix collects adapter data while the student is in control. This is a good extension.

**Remaining limit.** History cannot reveal a parameter that does not influence what you observe. Hence the exercise: which parameters does the student estimate well?

## Exercises
1. Print per-parameter adapter error. Which of the four is hardest?
2. Vary history length $K\in\{2,8,32\}$. How do estimation error and return change?
3. Implement the DAgger-style data collection and compare.

## Article seed
*"Teaching a policy to feel the world: privileged learning and online adaptation."*

**References:** Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots* (2021); Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain* (2020); Ross et al., *DAgger* (2011).
