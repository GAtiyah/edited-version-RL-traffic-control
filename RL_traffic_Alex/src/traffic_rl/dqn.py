"""Minimal DQN agent for the traffic control starter project."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Simple MLP used to estimate Q values."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Fixed-size replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states = torch.as_tensor(np.asarray([item[0] for item in batch]), dtype=torch.float32, device=device)
        actions = torch.as_tensor([item[1] for item in batch], dtype=torch.long, device=device)
        rewards = torch.as_tensor([item[2] for item in batch], dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.asarray([item[3] for item in batch]), dtype=torch.float32, device=device)
        dones = torch.as_tensor([item[4] for item in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)


@dataclass
class DQNConfig:
    """Hyperparameters for DQN."""

    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    hidden_dims: tuple[int, ...] = (128, 128)
    target_sync_steps: int = 250
    device: str = "cpu"


class DQNAgent:
    """DQN agent with experience replay and target network."""

    def __init__(self, observation_dim: int, action_dim: int, config: DQNConfig) -> None:
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.device)

        self.q_network = QNetwork(observation_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_network = QNetwork(observation_dim, action_dim, config.hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        self.training_steps = 0

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size=self.config.batch_size,
            device=self.device,
        )

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_values

        loss = F.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_steps += 1
        if self.training_steps % self.config.target_sync_steps == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
