"""Experience replay buffer for off-policy TD3 training."""

from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular replay buffer with vectorized random sampling."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
    ):
        self.capacity = int(capacity)
        self.device = device

        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size
