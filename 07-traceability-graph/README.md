# 07 · Traceability graph (PLM thinking for experiments)

**Goal:** record *which configuration, code version and training run produced each number*, in a graph you can query. This is the experiment-world version of lifecycle management.

## Run it
```bash
# 1. start Neo4j (Docker on CachyOS: sudo pacman -S docker && sudo systemctl enable --now docker)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/sim2real-pass neo4j:5
# 2. load results
uv pip install neo4j
python 07-traceability-graph/ingest.py
# 3. open http://localhost:7474 and run the queries in queries.cypher
```

## Graph model
```
(:Run {id, seed, git}) -[:OF_METHOD]-> (:Method {name, dr, steps, privileged, config_hash})
(:Run) -[:EVALUATED_ON {mean_return}]-> (:Condition {name, mass, friction, damping, gain, delay, obs_noise})
```

## The math (and the idea)

**Content addressing.** Give every configuration an identifier derived from its content,
$$h=H\big(\mathrm{canonical}(\text{config})\big),$$
where $H$ is a cryptographic hash and "canonical" means sorted keys and fixed formatting. Equal configs get equal ids, any change gives a new id, so a result can never silently refer to a changed setup.

**Lineage as a DAG.** Artifacts form a directed acyclic graph: config $\to$ run $\to$ evaluation $\to$ table cell. *Impact analysis* is reachability: if a node changes (for example the definition of the `real` condition), everything reachable downstream must be recomputed. Query 4 does exactly that.

**Reproducibility, defined.** A result is reproducible if (code commit, config hash, seed, environment lockfile) determines it up to documented randomness. Store those four things per run.

**Why a graph, not a table.** Questions such as "which results depend on this condition and were made with code older than commit X?" are multi-hop. In SQL that is several joins; in a graph it is one path pattern. Same reasoning as GraphRAG over engineering data.

## Exercises
1. Add `:Environment {lockfile_hash}` and link each run to it.
2. Write a query that lists methods whose `real` result differs by more than 20% across seeds.
3. Add a node for each figure in your articles and link it to the runs it uses.

**Not tested here:** this folder needs Docker and a running Neo4j, which I could not run in my sandbox. Syntax-checked only.

## Article seed
*"Provenance for ML experiments: a PLM-style traceability graph."*

**References:** Neo4j Cypher manual; W3C PROV overview; Pineau et al., *Improving Reproducibility in ML Research* (2021).
