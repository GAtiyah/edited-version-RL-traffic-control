# Notebook Status

No project notebooks are committed yet.

Recommended first notebooks after the runnable core is stable:

1. `01_simulator_sanity_check.ipynb`
   Validate queue growth, departures, and switching behavior under simple schedules.
2. `02_baseline_comparison.ipynb`
   Load `results/baseline_summary.json` and compare the heuristic controllers across regimes.
3. `03_dqn_training_analysis.ipynb`
   Load `results/dqn_summary.json`, inspect training history, and compare DQN against baselines.

Use the scripts first. Treat notebooks as analysis and presentation layers on top of the JSON outputs.
