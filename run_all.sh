#!/usr/bin/env bash
# Full pipeline. STEPS=300000 per run, 3 seeds. Roughly 1-2 hours on a laptop CPU.
set -euo pipefail
STEPS=${STEPS:-300000}
SEEDS=${SEEDS:-"0 1 2"}
export PYTHONPATH=src
mkdir -p models results

echo "== 04 system identification =="
python -m s2r.sysid
SIM=$(python -c "import json;d=json.load(open('results/sysid_params.json'))['estimated'];print(*d)")

for s in $SEEDS; do
  echo "== training seed $s =="
  python -m s2r.train --name base    --seed $s --steps $STEPS
  python -m s2r.train --name dr05    --seed $s --steps $STEPS --dr 0.5
  python -m s2r.train --name dr10    --seed $s --steps $STEPS --dr 1.0
  python -m s2r.train --name sysid   --seed $s --steps $STEPS --sim-physics $SIM
  python -m s2r.train --name teacher --seed $s --steps $STEPS --dr 1.0 --privileged
done

echo "== evaluation =="
for m in base dr05 dr10 sysid; do python -m s2r.eval_suite --method $m; done
python -m s2r.gap_sweep --model models/base_s0
python -m s2r.adapt --teacher models/teacher_s0
python -m s2r.stats | tee results/table.md
