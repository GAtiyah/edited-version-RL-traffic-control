#!/usr/bin/env python3
"""Evaluate baseline controllers across configured traffic regimes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.baselines import FixedCycleController, MaxPressureController, QueueThresholdController
from traffic_rl.config import build_env_kwargs, load_config
from traffic_rl.env import AdaptiveTrafficSignalEnv
from traffic_rl.evaluation import evaluate_policies


def format_metric(value: float) -> str:
    return f"{value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/baseline_summary.json",
        help="Path to write JSON summary.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    env_config = config["environment"]
    eval_config = config["evaluation"]

    policies = {
        "fixed_cycle": FixedCycleController(cycle_length=10),
        "queue_threshold": QueueThresholdController(threshold=5.0, min_green=3),
        "max_pressure": MaxPressureController(min_green=2),
    }

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    print("Evaluating baselines...\n")
    for regime_name, schedule in env_config["evaluation_regimes"].items():
        env_kwargs = build_env_kwargs(env_config, schedule)
        env_factory = lambda env_kwargs=env_kwargs: AdaptiveTrafficSignalEnv(**env_kwargs)
        results = evaluate_policies(
            env_factory=env_factory,
            policies=policies,
            episodes=int(eval_config.get("episodes_per_regime", 10)),
            base_seed=0,
        )
        all_results[regime_name] = results

        print(f"Regime: {regime_name}")
        for policy_name, summary in results.items():
            print(
                "  "
                f"{policy_name:16s} | "
                f"avg_queue={format_metric(summary['average_queue_length'])} | "
                f"avg_wait_s={format_metric(summary['average_wait_time_seconds'])} | "
                f"throughput={format_metric(summary['throughput_per_step'])} | "
                f"switches={format_metric(summary['switch_count'])}"
            )
        print()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(all_results, file, indent=2)

    print(f"Saved baseline summary to {output_path}")


if __name__ == "__main__":
    main()
