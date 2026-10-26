// 1. Best method per condition (by mean over seeds)
MATCH (r:Run)-[e:EVALUATED_ON]->(c:Condition), (r)-[:OF_METHOD]->(m:Method)
WITH c.name AS condition, m.name AS method, avg(e.mean_return) AS ret
ORDER BY condition, ret DESC
RETURN condition, collect({method: method, ret: round(ret)})[0] AS best;

// 2. The sim2real gap per method: nominal vs 'real'
MATCH (r:Run)-[a:EVALUATED_ON]->(:Condition {name:'nominal'}), (r)-[b:EVALUATED_ON]->(:Condition {name:'real'}),
      (r)-[:OF_METHOD]->(m:Method)
WITH m.name AS method, avg(a.mean_return) AS nominal, avg(b.mean_return) AS real
RETURN method, round(nominal) AS nominal, round(real) AS real,
       round(100.0 * (nominal - real) / nominal) AS gap_pct ORDER BY gap_pct;

// 3. Full provenance of one result: run -> method -> config hash -> git commit
MATCH (r:Run {id:'dr10_s0'})-[:OF_METHOD]->(m:Method)
RETURN r.id, r.git AS commit, m.name, m.config_hash, m.dr, m.steps, m.privileged;

// 4. Which runs were affected if a condition definition changes?
MATCH (c:Condition {name:'real'})<-[:EVALUATED_ON]-(r:Run)-[:OF_METHOD]->(m:Method)
RETURN m.name, count(r) AS runs_to_reevaluate;
