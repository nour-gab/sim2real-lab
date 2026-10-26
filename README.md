# sim2real-lab

A small, reproducible study of the **sim-to-real gap**, built as eight self-contained steps. Each folder has a `README.md` with the math and commands, and each README is the seed of a theoretical article.

The "real" system is a hidden, perturbed copy of the MuJoCo Hopper (different mass, friction, damping, actuator gain, delay and sensor noise). This is a **sim-to-sim proxy**: honest about not being hardware, and ideal for rigorous measurement.

| Step | Topic | Code |
|---|---|---|
| 01 | MDP, policy gradient, PPO | `src/s2r/train.py` |
| 02 | Define and measure the gap | `gap_sweep.py`, `eval_suite.py` |
| 03 | Domain randomization | `DRWrapper` in `envs.py` |
| 04 | System identification | `sysid.py` |
| 05 | Privileged teacher + adapter | `adapt.py` |
| 06 | Rigorous evaluation | `stats.py` |
| 07 | Traceability graph (Neo4j) | `07-traceability-graph/` |
| 08 | Governed agent | `08-governed-agent/` |

## Setup (CachyOS / Arch)
```bash
sudo pacman -S --needed git python uv base-devel
cd sim2real-lab
uv venv && source .venv/bin/activate
uv pip install -e .
python -c "import mujoco, gymnasium; print(mujoco.__version__, gymnasium.__version__)"
```
Everything runs on CPU. If you want a smaller PyTorch download: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` before the line above.

## Run everything
```bash
STEPS=300000 ./run_all.sh        # ~1-2 hours on a laptop CPU; use STEPS=100000 first
```

## Publishing the articles
Copy each folder's `README.md` into your GitHub Pages site (enable KaTeX or MathJax for the equations).
