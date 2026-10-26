# 02 · Measuring the sim2real gap

**Goal:** define the gap precisely, then find *which* parameter hurts most. The hidden "real" system is a perturbed Hopper (`src/s2r/conditions.py`), our stand-in for hardware.

## Run it
```bash
export PYTHONPATH=src
python -m s2r.gap_sweep --model models/base_s0 --episodes 8
# -> results/sweep.png and results/sweep.json
python -m s2r.eval_suite --method base      # -> results/base.json (all target conditions)
```

## The math

**Distribution shift view.** Training uses dynamics $P_{src}$; deployment uses $P_{tgt}$. The same policy has two returns, $J_{src}(\pi)$ and $J_{tgt}(\pi)$.

**Normalized gap.**
$$G(\pi)=\frac{J_{src}(\pi)-J_{tgt}(\pi)}{|J_{src}(\pi)|}\in(-\infty,1]$$
$G=0$ means no loss, $G=1$ means total loss. Normalizing makes tasks comparable.

**Sensitivity analysis.** For parameter $\theta_k$ perturb it to $\theta_k+\delta$ and estimate
$$S_k\approx\frac{J(\pi;\theta_k+\delta)-J(\pi;\theta_k)}{\delta}.$$
Rank parameters by $|S_k|\cdot\text{(plausible range of }\theta_k)$. A parameter with a steep curve *and* large real-world uncertainty is where effort pays off.

**Why delay is special.** With action delay $d$ the next state depends on $a_{t-d}$, which the policy's observation does not contain. The observation is no longer Markov: the effective state is $(s_t,a_{t-1},\dots,a_{t-d})$, i.e. a POMDP. Policies without memory cannot undo it, which is why delay often causes a sharp performance cliff.

**Diagnose, do not guess.** Replay the *same open-loop action sequence* in simulation and on the real system and compare responses; the first divergence points to the culprit module (actuation, contact, latency, perception). `sysid.py` collects exactly this kind of data.

## What to look for
Per-parameter curves in `sweep.png`, a ranking of sensitivities, and the gap of `base` on the `real` condition.

## Exercises
1. Which parameter has the steepest curve? Is that the one you expected?
2. Make `real` harder by adding delay 3. What happens to the gap?
3. Estimate $G$ with 3, 10 and 30 evaluation episodes. How noisy is it?

## Article seed
*"Measuring the sim2real gap: a sensitivity-first approach."*

**References:** Zhao et al., *Sim-to-Real Transfer in Deep RL for Robotics: a Survey* (2020); Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots* (2019).
