# 06 · Evaluating rigorously

**Goal:** turn runs into claims you can defend: seeds, robust aggregates, confidence intervals, a single results table.

## Run it
```bash
export PYTHONPATH=src
python -m s2r.stats | tee results/table.md      # table + results/real_comparison.png
```
`src/s2r/stats.py` reads every `results/<method>.json` (a list of per-seed mean returns for each condition).

## The math

**Why single runs lie.** Returns across seeds vary a lot in deep RL (Henderson et al., 2018). Reporting a best seed or a 3-seed mean without uncertainty is not evidence.

**Interquartile mean (IQM).** Sort values, drop the lowest and highest 25%, average the rest:
$$\mathrm{IQM}(x)=\frac{1}{n-2\lfloor n/4\rfloor}\sum_{i=\lfloor n/4\rfloor+1}^{n-\lfloor n/4\rfloor}x_{(i)}.$$
It is more robust to outliers than the mean and less wasteful than the median (Agarwal et al., 2021).

**Bootstrap confidence interval.** Resample the $n$ seed results with replacement $B$ times, compute the statistic each time, and take the $\alpha/2$ and $1-\alpha/2$ percentiles:
$$\mathrm{CI}_{1-\alpha}=\big[q_{\alpha/2},\ q_{1-\alpha/2}\big].$$
With only 3-5 seeds the CI is crude and optimistic; say so, and use more seeds for important claims.

**Comparing methods.** Overlapping CIs mean "not distinguishable at this sample size", not "equal". Prefer paired comparisons (same seeds and evaluation episodes) and report effect sizes. Control for multiple comparisons when you test many conditions.

**Reporting checklist.**
1. Same evaluation episodes and initial states for all methods.
2. $\ge 5$ seeds for headline claims.
3. Held-out conditions (`real` is never used for training or tuning).
4. Ablations: remove one component at a time.
5. A failure taxonomy, not only a mean.

## Exercises
1. Increase seeds to 5. How do the CIs change?
2. Add IQM to the table. When does it differ from the mean?
3. Write the claim your data supports in one honest sentence.

## Article seed
*"Reporting sim2real results honestly: seeds, IQM and bootstrap intervals."*

**References:** Henderson et al., *Deep RL That Matters* (2018); Agarwal et al., *Deep RL at the Edge of the Statistical Precipice* (2021).
