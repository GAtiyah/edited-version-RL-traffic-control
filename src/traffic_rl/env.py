"""Adaptive traffic signal control environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Mapping, Sequence

import numpy as np

DIRECTIONS = ("N", "S", "E", "W")
NS_DIRECTIONS = ("N", "S")
EW_DIRECTIONS = ("E", "W")

KEEP_ACTION = 0
SWITCH_ACTION = 1


@dataclass(frozen=True)
class ScheduleSegment:
    """Arrival-rate segment active until a given time step."""

    until_step: int
    rates: Dict[str, float]


@dataclass
class TrafficMetrics:
    """Episode-level metrics tracked by the environment."""

    total_reward: float = 0.0
    total_departed: int = 0
    total_wait_time_steps: float = 0.0
    switch_count: int = 0
    queue_history: list[int] = field(default_factory=list)
    reward_history: list[float] = field(default_factory=list)
    phase_history: list[int] = field(default_factory=list)
    throughput_history: list[int] = field(default_factory=list)

    def summary(self, step_seconds: int) -> dict[str, float]:
        """Aggregate metrics for reporting."""
        avg_queue = float(np.mean(self.queue_history)) if self.queue_history else 0.0
        max_queue = int(max(self.queue_history)) if self.queue_history else 0
        avg_wait_steps = (
            self.total_wait_time_steps / self.total_departed if self.total_departed else 0.0
        )
        throughput_per_step = (
            self.total_departed / len(self.queue_history) if self.queue_history else 0.0
        )

        return {
            "total_reward": float(self.total_reward),
            "average_queue_length": avg_queue,
            "maximum_queue_length": float(max_queue),
            "throughput_per_step": float(throughput_per_step),
            "total_departed": float(self.total_departed),
            "average_wait_time_steps": float(avg_wait_steps),
            "average_wait_time_seconds": float(avg_wait_steps * step_seconds),
            "switch_count": float(self.switch_count),
        }


class AdaptiveTrafficSignalEnv:
    """Simple single-intersection traffic signal control simulator."""

    def __init__(
        self,
        arrival_schedule: Sequence[Mapping[str, Any]],
        episode_length: int = 200,
        step_seconds: int = 3,
        yellow_time: int = 1,
        max_departures_per_step: int = 4,
        reward_mode: str = "queue",
        switch_penalty: float = 2.0,
        seed: int | None = None,
    ) -> None:
        if reward_mode not in {"queue", "waiting"}:
            raise ValueError("reward_mode must be either 'queue' or 'waiting'")

        self.arrival_schedule = self._normalize_schedule(arrival_schedule)
        self.episode_length = episode_length
        self.step_seconds = step_seconds
        self.yellow_time = yellow_time
        self.max_departures_per_step = max_departures_per_step
        self.reward_mode = reward_mode
        self.switch_penalty = switch_penalty
        self.rng = np.random.default_rng(seed)

        self.observation_dim = 10
        self.action_dim = 2

        self.step_count = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.switch_cooldown = 0
        self.queues: Dict[str, Deque[int]] = {}
        self.metrics = TrafficMetrics()

    @staticmethod
    def _normalize_schedule(
        schedule: Sequence[Mapping[str, Any]],
    ) -> list[ScheduleSegment]:
        if not schedule:
            raise ValueError("arrival_schedule must contain at least one segment")

        normalized: list[ScheduleSegment] = []
        previous_until = -1

        for segment in schedule:
            until_step = int(segment["until_step"])
            rates = {direction: float(segment["rates"].get(direction, 0.0)) for direction in DIRECTIONS}

            if until_step <= previous_until:
                raise ValueError("arrival_schedule must be sorted by increasing until_step")

            previous_until = until_step
            normalized.append(ScheduleSegment(until_step=until_step, rates=rates))

        return normalized

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the episode state and return the initial observation."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.current_phase = 0
        self.phase_duration = 0
        self.switch_cooldown = 0
        self.queues = {direction: deque() for direction in DIRECTIONS}
        self.metrics = TrafficMetrics()

        observation = self._get_observation()
        info = {
            "queue_length": 0,
            "arrival_rates": self._current_arrival_rates(),
            "phase": self.current_phase,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Advance the simulator by one step."""
        if action not in {KEEP_ACTION, SWITCH_ACTION}:
            raise ValueError("action must be 0 (keep) or 1 (switch)")
        if self.step_count >= self.episode_length:
            raise RuntimeError("episode is done; call reset() before calling step() again")

        switch_this_step = action == SWITCH_ACTION and self.switch_cooldown == 0
        if switch_this_step:
            self.current_phase = 1 - self.current_phase
            self.phase_duration = 0
            self.switch_cooldown = self.yellow_time
            self.metrics.switch_count += 1

        waiting_increment = sum(len(queue) for queue in self.queues.values())
        self._age_queued_vehicles()

        arrivals = self._sample_arrivals()
        for direction, count in arrivals.items():
            self.queues[direction].extend([0] * count)

        departures = {direction: 0 for direction in DIRECTIONS}

        if self.switch_cooldown == 0:
            active_directions = NS_DIRECTIONS if self.current_phase == 0 else EW_DIRECTIONS
            for direction in active_directions:
                departures[direction] = min(len(self.queues[direction]), self.max_departures_per_step)
                for _ in range(departures[direction]):
                    self.metrics.total_wait_time_steps += self.queues[direction].popleft()
                    self.metrics.total_departed += 1
            self.phase_duration += 1
        else:
            self.switch_cooldown -= 1

        queue_length = sum(len(queue) for queue in self.queues.values())
        reward_signal = -queue_length if self.reward_mode == "queue" else -waiting_increment
        reward = float(reward_signal - (self.switch_penalty if switch_this_step else 0.0))

        self.metrics.total_reward += reward
        self.metrics.queue_history.append(queue_length)
        self.metrics.reward_history.append(reward)
        self.metrics.phase_history.append(self.current_phase)
        self.metrics.throughput_history.append(sum(departures.values()))

        self.step_count += 1
        done = self.step_count >= self.episode_length

        observation = self._get_observation()
        info = {
            "step": self.step_count,
            "queue_length": queue_length,
            "arrivals": arrivals,
            "departures": departures,
            "phase": self.current_phase,
            "phase_duration": self.phase_duration,
            "switch_cooldown": self.switch_cooldown,
        }
        return observation, reward, done, info

    def summarize(self) -> dict[str, float]:
        """Return a summary of the current episode metrics."""
        return self.metrics.summary(step_seconds=self.step_seconds)

    def _current_arrival_rates(self) -> dict[str, float]:
        for segment in self.arrival_schedule:
            if self.step_count < segment.until_step:
                return dict(segment.rates)
        return dict(self.arrival_schedule[-1].rates)

    def _sample_arrivals(self) -> dict[str, int]:
        rates = self._current_arrival_rates()
        return {
            direction: int(self.rng.poisson(rates[direction]))
            for direction in DIRECTIONS
        }

    def _age_queued_vehicles(self) -> None:
        for direction in DIRECTIONS:
            self.queues[direction] = deque(age + 1 for age in self.queues[direction])

    def _get_observation(self) -> np.ndarray:
        arrival_rates = self._current_arrival_rates()
        values = [
            len(self.queues.get("N", [])),
            len(self.queues.get("S", [])),
            len(self.queues.get("E", [])),
            len(self.queues.get("W", [])),
            self.current_phase,
            self.phase_duration,
            arrival_rates["N"],
            arrival_rates["S"],
            arrival_rates["E"],
            arrival_rates["W"],
        ]
        return np.asarray(values, dtype=np.float32)
