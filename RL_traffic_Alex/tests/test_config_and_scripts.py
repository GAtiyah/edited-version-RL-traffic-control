"""Config loading and script smoke tests."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from traffic_rl.config import load_config


SMOKE_CONFIG = """
environment:
  episode_length: 12
  step_seconds: 3
  yellow_time: 1
  max_departures_per_step: 3
  reward_mode: queue
  switch_penalty: 1.0
  train_schedule:
    - until_step: 12
      rates:
        N: 0.5
        S: 0.5
        E: 0.7
        W: 0.7
  evaluation_regimes:
    symmetric:
      - until_step: 12
        rates:
          N: 0.4
          S: 0.4
          E: 0.4
          W: 0.4
    asymmetric:
      - until_step: 12
        rates:
          N: 0.3
          S: 0.3
          E: 0.9
          W: 0.9

training:
  episodes: 2
  gamma: 0.95
  learning_rate: 0.001
  batch_size: 4
  buffer_size: 128
  hidden_dims: [16, 16]
  start_epsilon: 0.8
  end_epsilon: 0.1
  epsilon_decay_steps: 20
  warmup_steps: 2
  update_frequency: 1
  target_sync_steps: 4
  seed: 3

evaluation:
  episodes_per_regime: 2
"""


class ConfigAndScriptSmokeTest(unittest.TestCase):
    def test_load_config_without_pyyaml_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            config = load_config(config_path)

        self.assertEqual(config["environment"]["episode_length"], 12)
        self.assertEqual(config["training"]["hidden_dims"], [16, 16])
        self.assertEqual(config["evaluation"]["episodes_per_regime"], 2)

    def test_run_baselines_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            output_path = Path(tmpdir) / "baseline_summary.json"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "run_baselines.py"),
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertIn("symmetric", payload)
        self.assertIn("max_pressure", payload["symmetric"])
        self.assertIn("Saved baseline summary", result.stdout)

    def test_run_training_script_and_summary_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            checkpoint_path = Path(tmpdir) / "checkpoints" / "policy.pt"
            summary_path = Path(tmpdir) / "dqn_summary.json"
            config_path.write_text(SMOKE_CONFIG, encoding="utf-8")

            train_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_dqn.py"),
                    "--config",
                    str(config_path),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--summary-output",
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            render_result = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "summarize_results.py"),
                    str(summary_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )

            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoint_exists = checkpoint_path.exists()

        self.assertTrue(checkpoint_exists)
        self.assertEqual(len(payload["training_history"]), 2)
        self.assertIn("dqn", payload["evaluation_results"]["symmetric"])
        self.assertIn("Saved checkpoint", train_result.stdout)
        self.assertIn("DQN summary", render_result.stdout)


if __name__ == "__main__":
    unittest.main()
