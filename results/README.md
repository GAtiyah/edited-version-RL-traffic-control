# Results

This directory stores generated experiment artifacts.

Expected outputs:

- `baseline_summary.json` from `scripts/run_baselines.py`
- `dqn_summary.json` from `scripts/train_dqn.py`
- `checkpoints/dqn_policy.pt` from `scripts/train_dqn.py`

Quick inspection:

```bash
python3 scripts/summarize_results.py results/baseline_summary.json
python3 scripts/summarize_results.py results/dqn_summary.json
```
