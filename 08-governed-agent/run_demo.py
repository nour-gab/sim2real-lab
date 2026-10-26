"""python run_demo.py [--claude] [--approve]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent import Agent, ClaudePlanner, RulePlanner, ask_human, deny_all

ap = argparse.ArgumentParser()
ap.add_argument("--claude", action="store_true")
ap.add_argument("--approve", action="store_true", help="ask a human for write approvals")
a = ap.parse_args()
agent = Agent(ClaudePlanner() if a.claude else RulePlanner(),
              ask_human if a.approve else deny_all, trace="trace.jsonl")
for o in agent.run("Compare all methods on the 'real' condition and say which is best."):
    print(o)
