"""
Quick sanity-check: confirms the direct-answer bypass is loaded and working.
Run this from the engine/ directory:
    python test_direct_answer.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from physicsflow.agent.context_provider import ReservoirContextProvider
from physicsflow.agent.reservoir_agent import ReservoirAgent

print("Loading agent...")
ctx   = ReservoirContextProvider()
ctx._seed_norne_baseline()
agent = ReservoirAgent(context_provider=ctx)

# Check method exists
has_method = hasattr(agent, '_try_direct_answer')
print(f"_try_direct_answer exists: {has_method}")
if not has_method:
    print("FAIL — old code is loaded. Run: pip install -e . then retry.")
    sys.exit(1)

# Run a real data question
msg = "Which wells are performing above and below expectations?"
gen = agent._try_direct_answer("test", msg)
if gen is None:
    print("FAIL — method returned None (keyword not matched).")
    sys.exit(1)

full = ""
for chunk in gen:
    full += chunk.get("token", "")

if "above expectation" in full.lower() or "below expectation" in full.lower():
    print("\nPASS — direct answer fired correctly:\n")
    print(full)
else:
    print(f"\nWARN — response doesn't look like well data:\n{full}")
