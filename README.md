# Reinforcement Learning for Adaptive Traffic Signal Control

This repository is a course-project starter for adaptive traffic signal control at a single intersection under stationary and nonstationary demand.

## Status

Implemented now:

- single-intersection simulator with stochastic arrivals
- three heuristic baselines: fixed-cycle, queue-threshold, max-pressure
- DQN training loop with replay buffer and target network
- JSON result outputs for baselines and DQN training/evaluation
- smoke tests for the environment and the main scripts

Planned but not included yet:

- finished analysis notebooks
- presentation-ready plots and figures
- checkpoint resume / experiment tracking
- multi-intersection extensions

## Project Goal

The controller chooses whether to keep or switch the traffic-light phase at each step. The objective is to reduce congestion and waiting time while accounting for switching costs.

Core question:

Can an RL policy learn a better long-horizon controller than fixed-cycle and queue-based heuristics, especially when traffic demand changes over time?

## Repository Layout

```text
RL_traffic_Alex/
├── configs/
│   └── default.yaml
├── docs/
│   └── proposal_draft.md
├── notebooks/
│   └── README.md
├── results/
├── scripts/
│   ├── run_baselines.py
│   ├── summarize_results.py
│   └── train_dqn.py
├── src/
│   └── traffic_rl/
│       ├── __init__.py
│       ├── baselines.py
│       ├── config.py
│       ├── dqn.py
│       ├── env.py
│       └── evaluation.py
├── tests/
│   ├── test_config_and_scripts.py
│   └── test_env.py
├── requirements.txt
└── requirements-optional.txt
```

## Environment Summary

- phase `0`: north-south green
- phase `1`: east-west green
- action `0`: keep current phase
- action `1`: switch phase
- default observation:
  - queue lengths for `N, S, E, W`
  - current phase
  - current phase duration
  - current arrival rates for `N, S, E, W`

The simulator includes:

- Poisson arrivals from configurable piecewise demand regimes
- per-step departure capacity
- yellow-time switch loss
- queue-based or waiting-based reward shaping

## Setup

Core runtime:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional extras for notebooks, plotting, or alternative YAML parsing:

```bash
pip install -r requirements-optional.txt
```

`PyYAML` is optional now. The project can read the included config files without it.

## Verification

Run the tests:

```bash
python3 -m unittest discover -s tests
```

Run baseline evaluation:

```bash
python3 scripts/run_baselines.py --config configs/default.yaml
python3 scripts/summarize_results.py results/baseline_summary.json
```

Train and evaluate DQN:

```bash
python3 scripts/train_dqn.py --config configs/default.yaml
python3 scripts/summarize_results.py results/dqn_summary.json
```

## Outputs

Main generated artifacts:

- `results/baseline_summary.json`
- `results/dqn_summary.json`
- `results/checkpoints/dqn_policy.pt`

Reported metrics:

- `total_reward`
- `average_queue_length`
- `maximum_queue_length`
- `throughput_per_step`
- `total_departed`
- `average_wait_time_steps`
- `average_wait_time_seconds`
- `switch_count`

## Known Limitations

- only a single intersection is modeled
- demand is synthetic rather than data-driven
- there is no checkpoint resume path yet
- notebooks are still placeholders
- experiment tracking is minimal and file-based

## Recommended Next Steps

1. Run the baselines and DQN pipeline once end to end.
2. Compare evaluation metrics across the configured regimes.
3. Use `scripts/summarize_results.py` plus a notebook or report table for presentation.
4. Add ablations for reward design, state representation, and switch penalty if time allows.
