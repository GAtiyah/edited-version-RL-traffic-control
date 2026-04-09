"""Evaluation helpers for traffic control policies."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .env import AdaptiveTrafficSignalEnv

Policy = Callable[[np.ndarray], int]


def _resolve_action(policy: Any, observation: np.ndarray) -> int:
    if hasattr(policy, "act"):
        return int(policy.act(observation))
    return int(policy(observation))


def run_episode(
    env: AdaptiveTrafficSignalEnv,
    policy: Any,
    seed: int | None = None,
) -> dict[str, float]:
    """Run one episode and return summary metrics."""
    observation, _ = env.reset(seed=seed)
    done = False

    while not done:
        action = _resolve_action(policy, observation)
        observation, _, done, _ = env.step(action)

    return env.summarize()


def evaluate_policy(
    env_factory: Callable[[], AdaptiveTrafficSignalEnv],
    policy: Any,
    episodes: int = 10,
    base_seed: int = 0,
) -> dict[str, float]:
    """Evaluate one policy over multiple episodes."""
    episode_summaries = []
    for episode_idx in range(episodes):
        env = env_factory()
        summary = run_episode(env, policy, seed=base_seed + episode_idx)
        episode_summaries.append(summary)

    metrics = {}
    for key in episode_summaries[0]:
        metrics[key] = float(np.mean([summary[key] for summary in episode_summaries]))
    return metrics


def evaluate_policies(
    env_factory: Callable[[], AdaptiveTrafficSignalEnv],
    policies: Mapping[str, Any],
    episodes: int = 10,
    base_seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Evaluate many policies under the same environment factory."""
    return {
        name: evaluate_policy(env_factory, policy, episodes=episodes, base_seed=base_seed)
        for name, policy in policies.items()
    }
