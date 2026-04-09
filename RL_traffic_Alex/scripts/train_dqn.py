#!/usr/bin/env python3
"""Train a DQN controller and compare it against baseline policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.baselines import FixedCycleController, MaxPressureController, QueueThresholdController
from traffic_rl.config import build_env_kwargs, load_config
from traffic_rl.dqn import DQNAgent, DQNConfig
from traffic_rl.env import AdaptiveTrafficSignalEnv
from traffic_rl.evaluation import evaluate_policies


def linear_epsilon(global_step: int, start: float, end: float, decay_steps: int) -> float:
    if global_step >= decay_steps:
        return end
    fraction = global_step / max(decay_steps, 1)
    return start + fraction * (end - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/checkpoints/dqn_policy.pt",
        help="Path to save the trained model.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default="results/dqn_summary.json",
        help="Path to save training and evaluation metrics.",
    )
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / args.config)
    env_config = config["environment"]
    train_config = config["training"]
    eval_config = config["evaluation"]

    train_env = AdaptiveTrafficSignalEnv(
        **build_env_kwargs(env_config, env_config["train_schedule"]),
        seed=int(train_config.get("seed", 0)),
    )

    agent = DQNAgent(
        observation_dim=train_env.observation_dim,
        action_dim=train_env.action_dim,
        config=DQNConfig(
            gamma=float(train_config.get("gamma", 0.99)),
            learning_rate=float(train_config.get("learning_rate", 1e-3)),
            batch_size=int(train_config.get("batch_size", 64)),
            buffer_size=int(train_config.get("buffer_size", 50_000)),
            hidden_dims=tuple(train_config.get("hidden_dims", [128, 128])),
            target_sync_steps=int(train_config.get("target_sync_steps", 250)),
            device=str(train_config.get("device", "cpu")),
        ),
    )

    global_step = 0
    training_history = []
    episodes = int(train_config.get("episodes", 250))
    warmup_steps = int(train_config.get("warmup_steps", 500))
    update_frequency = int(train_config.get("update_frequency", 1))
    start_epsilon = float(train_config.get("start_epsilon", 1.0))
    end_epsilon = float(train_config.get("end_epsilon", 0.05))
    epsilon_decay_steps = int(train_config.get("epsilon_decay_steps", 20_000))
    base_seed = int(train_config.get("seed", 0))
    epsilon = end_epsilon

    print("Training DQN...\n")
    for episode_idx in range(episodes):
        observation, _ = train_env.reset(seed=base_seed + episode_idx)
        done = False
        losses = []

        while not done:
            epsilon = linear_epsilon(global_step, start_epsilon, end_epsilon, epsilon_decay_steps)
            action = agent.act(observation, epsilon=epsilon)
            next_observation, reward, done, _ = train_env.step(action)
            agent.observe(observation, action, reward, next_observation, done)

            if global_step >= warmup_steps and global_step % update_frequency == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            observation = next_observation
            global_step += 1

        summary = train_env.summarize()
        summary["episode"] = float(episode_idx)
        summary["epsilon"] = float(epsilon)
        if losses:
            summary["mean_loss"] = float(sum(losses) / len(losses))
        training_history.append(summary)

        if (episode_idx + 1) % 25 == 0 or episode_idx == 0:
            print(
                f"Episode {episode_idx + 1:4d}/{episodes} | "
                f"reward={summary['total_reward']:.2f} | "
                f"avg_queue={summary['average_queue_length']:.2f} | "
                f"avg_wait_s={summary['average_wait_time_seconds']:.2f} | "
                f"epsilon={summary['epsilon']:.3f}"
            )

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(checkpoint_path))

    policies = {
        "fixed_cycle": FixedCycleController(cycle_length=10),
        "queue_threshold": QueueThresholdController(threshold=5.0, min_green=3),
        "max_pressure": MaxPressureController(min_green=2),
        "dqn": lambda observation: agent.act(observation, epsilon=0.0),
    }

    evaluation_results: dict[str, dict[str, dict[str, float]]] = {}
    print("\nEvaluating trained DQN...\n")
    for regime_name, schedule in env_config["evaluation_regimes"].items():
        env_kwargs = build_env_kwargs(env_config, schedule)
        env_factory = lambda env_kwargs=env_kwargs: AdaptiveTrafficSignalEnv(**env_kwargs)
        regime_results = evaluate_policies(
            env_factory=env_factory,
            policies=policies,
            episodes=int(eval_config.get("episodes_per_regime", 10)),
            base_seed=10_000,
        )
        evaluation_results[regime_name] = regime_results

        dqn_summary = regime_results["dqn"]
        print(
            f"{regime_name:18s} | "
            f"avg_queue={dqn_summary['average_queue_length']:.2f} | "
            f"avg_wait_s={dqn_summary['average_wait_time_seconds']:.2f} | "
            f"throughput={dqn_summary['throughput_per_step']:.2f}"
        )

    output_path = PROJECT_ROOT / args.summary_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training_history": training_history,
                "evaluation_results": evaluation_results,
                "checkpoint": str(checkpoint_path),
            },
            file,
            indent=2,
        )

    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Saved training summary to {output_path}")


if __name__ == "__main__":
    main()
