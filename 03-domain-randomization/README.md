# 03 · Domain randomization

**Goal:** make the policy robust by training on a *distribution* of simulators, then measure the robustness-versus-performance trade-off.

## Run it
```bash
export PYTHONPATH=src
for s in 0 1 2; do
  python -m s2r.train --name dr05 --seed $s --steps 300000 --dr 0.5
  python -m s2r.train --name dr10 --seed $s --steps 300000 --dr 1.0
done
for m in dr05 dr10; do python -m s2r.eval_suite --method $m; done
```
`DRWrapper` in `src/s2r/envs.py` resamples mass, friction, damping, actuator gain, delay and observation noise at every episode reset. Scale $s$ shrinks or widens the ranges around the nominal value.

## The math

**Objective.** Instead of one simulator $\theta_0$, optimize the expected return over a randomization distribution $p_s(\theta)$:
$$J_{DR}(\pi)=\mathbb E_{\theta\sim p_s}\,\mathbb E_{\tau\sim\pi,\theta}\Big[\sum_t\gamma^t r_t\Big].$$
If the real parameters $\theta^\star$ lie in the support of $p_s$ the real system looks like "just another sample" (Tobin et al., 2017).

**Worst case instead of average.** A robust alternative is $\max_\pi\min_{\theta\in\Theta}J(\pi;\theta)$. It is more conservative; the average is easier to optimize. Domain randomization sits in between, controlled by how wide $p_s$ is.

**Trade-off.** Let $\Delta(s)$ be the width of the distribution. Wider $\Delta$ increases coverage (the probability that $\theta^\star$ is in the support rises) but also increases the *variance* of the learning signal and pushes the policy toward behavior that is safe everywhere and optimal nowhere. Expect a curve: performance at nominal falls as $s$ grows, performance on `real` first rises, then falls.

**Automatic domain randomization (ADR).** Start with $s\approx 0$ and widen a boundary of $p_s$ whenever the policy's success there exceeds a threshold. This builds a curriculum and avoids guessing ranges (OpenAI, 2019).

**Pitfalls.** Unphysical combinations, ranges not tied to measurements, randomizing things that do not matter (wasted capacity), and forgetting that a *memoryless* policy cannot adapt: it averages over $\theta$. Section 05 fixes that with history.

## Exercises
1. Plot nominal return and `real` return against $s\in\{0,0.25,0.5,0.75,1\}$ (3 seeds each).
2. Randomize only gain and delay. Does that recover most of the benefit?
3. Narrow the ranges around the values estimated in section 04. Is that better than wide ranges?

## Article seed
*"Domain randomization is a prior, not magic: measuring the robustness-performance trade-off."*

**References:** Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World* (2017); Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization* (2018); OpenAI, *Solving Rubik's Cube with a Robot Hand* (2019).
