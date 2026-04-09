"""Baseline controllers for adaptive traffic signal control."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import KEEP_ACTION, SWITCH_ACTION


def _extract_state(observation: np.ndarray) -> dict[str, float]:
    return {
        "q_n": float(observation[0]),
        "q_s": float(observation[1]),
        "q_e": float(observation[2]),
        "q_w": float(observation[3]),
        "phase": int(observation[4]),
        "phase_duration": float(observation[5]),
    }


@dataclass
class FixedCycleController:
    """Switch every fixed number of active green steps."""

    cycle_length: int = 10
    name: str = "fixed_cycle"

    def act(self, observation: np.ndarray) -> int:
        state = _extract_state(observation)
        if state["phase_duration"] >= self.cycle_length:
            return SWITCH_ACTION
        return KEEP_ACTION


@dataclass
class QueueThresholdController:
    """Switch if the opposite direction is much more congested."""

    threshold: float = 5.0
    min_green: int = 3
    name: str = "queue_threshold"

    def act(self, observation: np.ndarray) -> int:
        state = _extract_state(observation)
        ns_queue = state["q_n"] + state["q_s"]
        ew_queue = state["q_e"] + state["q_w"]

        if state["phase_duration"] < self.min_green:
            return KEEP_ACTION

        if state["phase"] == 0 and (ew_queue - ns_queue) > self.threshold:
            return SWITCH_ACTION
        if state["phase"] == 1 and (ns_queue - ew_queue) > self.threshold:
            return SWITCH_ACTION
        return KEEP_ACTION


@dataclass
class MaxPressureController:
    """Always favor the larger queue once minimum green is satisfied."""

    min_green: int = 2
    name: str = "max_pressure"

    def act(self, observation: np.ndarray) -> int:
        state = _extract_state(observation)
        ns_queue = state["q_n"] + state["q_s"]
        ew_queue = state["q_e"] + state["q_w"]

        if state["phase_duration"] < self.min_green:
            return KEEP_ACTION

        if state["phase"] == 0 and ew_queue > ns_queue:
            return SWITCH_ACTION
        if state["phase"] == 1 and ns_queue > ew_queue:
            return SWITCH_ACTION
        return KEEP_ACTION
