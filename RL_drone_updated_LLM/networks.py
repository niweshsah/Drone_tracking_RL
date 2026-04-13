"""Neural network modules for TD3 (actor + twin critics)."""

from typing import Tuple

import torch
import torch.nn as nn


def init_linear(layer: nn.Linear) -> None:
    """Initialize linear layers with stable defaults for TD3 MLPs."""
    nn.init.kaiming_uniform_(layer.weight, a=0.0, nonlinearity="relu")
    nn.init.constant_(layer.bias, 0.0)


class Actor(nn.Module):
    """Policy network mapping a 24D stacked state to a 2D bounded action."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_linear(module)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Critic(nn.Module):
    """Single critic network that estimates Q(s, a)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init_linear(module)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=-1)
        return self.net(sa)


class TwinCritic(nn.Module):
    """Two independent critics used by TD3 to mitigate overestimation bias."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dim)
        self.q2 = Critic(state_dim, action_dim, hidden_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Convenience method for actor loss where only Q1 is required."""
        return self.q1(state, action)
