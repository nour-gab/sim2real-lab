"""Load experiment results into Neo4j with provenance: config hash, git commit, per-seed runs.

    docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/sim2real-pass neo4j:5
    pip install neo4j && python ingest.py
"""
import glob
import hashlib
import json
import subprocess
from pathlib import Path

from neo4j import GraphDatabase

URI, AUTH = "bolt://localhost:7687", ("neo4j", "sim2real-pass")
ROOT = Path(__file__).resolve().parent.parent
METHODS = {   # how each method was produced (edit to match what you ran)
    "base":            dict(dr=0.0, steps=300_000, privileged=False),
    "dr05":            dict(dr=0.5, steps=300_000, privileged=False),
    "dr10":            dict(dr=1.0, steps=300_000, privileged=False),
    "sysid":           dict(dr=0.0, steps=300_000, privileged=False),
    "teacher_oracle":  dict(dr=1.0, steps=300_000, privileged=True),
    "student":         dict(dr=1.0, steps=300_000, privileged=True),
}


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def main():
    from s2r.conditions import TARGETS
    commit = git_commit()
    driver = GraphDatabase.driver(URI, auth=AUTH)
    with driver.session() as s:
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE")
        s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Run) REQUIRE r.id IS UNIQUE")
        for name, p in TARGETS.items():
            s.run("MERGE (c:Condition {name:$n}) SET c += $p", n=name, p=vars(p))
        for f in sorted(glob.glob(str(ROOT / "results" / "*.json"))):
            method = Path(f).stem
            data = json.loads(Path(f).read_text())
            if method not in METHODS or "real" not in data:
                continue
            cfg = METHODS[method]
            s.run("MERGE (m:Method {name:$n}) SET m += $cfg, m.config_hash=$h",
                  n=method, cfg=cfg, h=config_hash(cfg))
            for cond, per_seed in data.items():
                for seed, ret in enumerate(per_seed):
                    s.run("""MATCH (m:Method {name:$m}), (c:Condition {name:$c})
                             MERGE (r:Run {id:$id}) SET r.seed=$seed, r.git=$git
                             MERGE (r)-[:OF_METHOD]->(m)
                             MERGE (r)-[e:EVALUATED_ON]->(c) SET e.mean_return=$ret""",
                          m=method, c=cond, id=f"{method}_s{seed}", seed=seed, git=commit, ret=ret)
    driver.close()
    print("ingested; open http://localhost:7474 and try queries.cypher")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    main()
