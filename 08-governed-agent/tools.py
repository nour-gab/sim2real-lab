"""Tool registry: schemas, risk levels, costs. Everything an agent may do is listed here."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

RESULTS = Path(__file__).resolve().parent.parent / "results"
NAME = r"^[a-z0-9_]+$"


class ListMethods(BaseModel):
    pass


class QueryResults(BaseModel):
    method: str = Field(pattern=NAME)
    condition: str = Field(pattern=NAME)


class LaunchTraining(BaseModel):
    method: str = Field(pattern=NAME)
    steps: int = Field(ge=10_000, le=1_000_000)       # hard schema limit
    dr: float = Field(ge=0.0, le=1.0)
    seed: int = Field(ge=0, le=99)


def list_methods(a: ListMethods):
    out = []
    for p in RESULTS.glob("*.json"):
        d = json.loads(p.read_text())
        if isinstance(d, dict) and "real" in d:
            out.append(p.stem)
    return sorted(out)


def query_results(a: QueryResults):
    d = json.loads((RESULTS / f"{a.method}.json").read_text())
    vals = d[a.condition]
    return {"method": a.method, "condition": a.condition,
            "mean_return": sum(vals) / len(vals), "n_seeds": len(vals), "note": d.get("note", "")}


def launch_training(a: LaunchTraining):
    cmd = (f"python -m s2r.train --name {a.method} --seed {a.seed} "
           f"--steps {a.steps} --dr {a.dr}")
    return {"dry_run": True, "command": cmd}           # execution is a separate, human decision


TOOLS = {
    "list_methods":    dict(model=ListMethods, fn=list_methods, risk="read",
                            doc="List methods that have evaluation results."),
    "query_results":   dict(model=QueryResults, fn=query_results, risk="read",
                            doc="Mean return of a method on a target condition."),
    "launch_training": dict(model=LaunchTraining, fn=launch_training, risk="write",
                            doc="Propose a PPO training run. Costs compute; needs human approval."),
}
