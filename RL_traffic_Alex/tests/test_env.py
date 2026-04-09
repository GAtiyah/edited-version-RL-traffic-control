"""Smoke tests for the traffic RL starter project."""

from __future__ import annotations

from pathlib import Path
import tempfile
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.baselines import FixedCycleController, MaxPressureController
from traffic_rl.dqn import DQNAgent, DQNConfig
from traffic_rl.env import AdaptiveTrafficSignalEnv, KEEP_ACTION, SWITCH_ACTION


ZERO_SCHEDULE = [
    {
        "until_step": 5,
        "rates": {"N": 0.0, "S": 0.0, "E": 0.0, "W": 0.0},
    }
]


class AdaptiveTrafficSignalEnvTest(unittest.TestCase):
    def test_invalid_schedule_order_raises(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTrafficSignalEnv(
                arrival_schedule=[
                    {"until_step": 5, "rates": {"N": 0.1, "S": 0.1, "E": 0.1, "W": 0.1}},
                    {"until_step": 4, "rates": {"N": 0.2, "S": 0.2, "E": 0.2, "W": 0.2}},
                ]
            )

    def test_invalid_reward_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveTrafficSignalEnv(arrival_schedule=ZERO_SCHEDULE, reward_mode="bad-mode")

    def test_reset_returns_expected_observation_shape(self) -> None:
        env = AdaptiveTrafficSignalEnv(arrival_schedule=ZERO_SCHEDULE, episode_length=5)
        observation, info = env.reset(seed=0)

        self.assertEqual(observation.shape, (10,))
        self.assertEqual(info["queue_length"], 0)
        self.assertEqual(info["phase"], 0)

    def test_episode_finishes_after_expected_steps(self) -> None:
        env = AdaptiveTrafficSignalEnv(
            arrival_schedule=[
                {
                    "until_step": 3,
                    "rates": {"N": 1.0, "S": 0.0, "E": 0.0, "W": 0.0},
                }
            ],
            episode_length=3,
        )

        env.reset(seed=0)
        done = False
        steps = 0

        while not done:
            _, _, done, _ = env.step(KEEP_ACTION)
            steps += 1

        summary = env.summarize()
        self.assertEqual(steps, 3)
        self.assertIn("average_queue_length", summary)
        self.assertGreaterEqual(summary["maximum_queue_length"], 0.0)

    def test_switch_action_changes_phase(self) -> None:
        env = AdaptiveTrafficSignalEnv(
            arrival_schedule=ZERO_SCHEDULE,
            episode_length=2,
            yellow_time=1,
        )

        env.reset(seed=0)
        self.assertEqual(env.current_phase, 0)
        _, _, _, info = env.step(SWITCH_ACTION)

        self.assertEqual(env.current_phase, 1)
        self.assertEqual(info["switch_cooldown"], 0)

    def test_step_after_done_raises(self) -> None:
        env = AdaptiveTrafficSignalEnv(arrival_schedule=ZERO_SCHEDULE, episode_length=1)
        env.reset(seed=0)
        env.step(KEEP_ACTION)

        with self.assertRaises(RuntimeError):
            env.step(KEEP_ACTION)

    def test_reset_with_same_seed_is_deterministic(self) -> None:
        env = AdaptiveTrafficSignalEnv(
            arrival_schedule=[
                {
                    "until_step": 4,
                    "rates": {"N": 0.9, "S": 0.1, "E": 0.4, "W": 0.2},
                }
            ],
            episode_length=4,
        )

        first_obs, _ = env.reset(seed=123)
        first_rollout = [env.step(KEEP_ACTION) for _ in range(4)]
        second_obs, _ = env.reset(seed=123)
        second_rollout = [env.step(KEEP_ACTION) for _ in range(4)]

        np.testing.assert_array_equal(first_obs, second_obs)
        self.assertEqual(
            [(reward, done, info["arrivals"]) for _, reward, done, info in first_rollout],
            [(reward, done, info["arrivals"]) for _, reward, done, info in second_rollout],
        )

    def test_fixed_cycle_switches_when_cycle_reached(self) -> None:
        controller = FixedCycleController(cycle_length=3)
        observation = np.asarray([0, 0, 0, 0, 0, 3, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(controller.act(observation), SWITCH_ACTION)

    def test_max_pressure_prefers_busier_direction(self) -> None:
        controller = MaxPressureController(min_green=2)
        observation = np.asarray([1, 1, 5, 5, 0, 4, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.assertEqual(controller.act(observation), SWITCH_ACTION)

    def test_dqn_checkpoint_round_trip(self) -> None:
        env = AdaptiveTrafficSignalEnv(arrival_schedule=ZERO_SCHEDULE, episode_length=6, seed=0)
        observation, _ = env.reset(seed=0)
        agent = DQNAgent(
            observation_dim=env.observation_dim,
            action_dim=env.action_dim,
            config=DQNConfig(batch_size=2, buffer_size=16, hidden_dims=(8, 8), target_sync_steps=2),
        )

        done = False
        while not done:
            action = agent.act(observation, epsilon=0.2)
            next_observation, reward, done, _ = env.step(action)
            agent.observe(observation, action, reward, next_observation, done)
            agent.update()
            observation = next_observation

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy.pt"
            agent.save(str(checkpoint_path))

            loaded_agent = DQNAgent(
                observation_dim=env.observation_dim,
                action_dim=env.action_dim,
                config=DQNConfig(batch_size=2, buffer_size=16, hidden_dims=(8, 8), target_sync_steps=2),
            )
            loaded_agent.load(str(checkpoint_path))

            sample_state = np.zeros(env.observation_dim, dtype=np.float32)

            with self.subTest("checkpoint_exists"):
                self.assertTrue(checkpoint_path.exists())

            with self.subTest("matching_policy_output"):
                sample_tensor = torch.as_tensor(sample_state, dtype=torch.float32).unsqueeze(0)
                original_q = agent.q_network(sample_tensor)
                loaded_q = loaded_agent.q_network(sample_tensor)
                self.assertTrue(torch.allclose(original_q, loaded_q))


if __name__ == "__main__":
    unittest.main()
