"""Twin Delayed Deep Deterministic Policy Gradient (TD3) agent."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from config import TD3Config
from networks import Actor, TwinCritic
from replay_buffer import ReplayBuffer


class TD3Agent:
    """TD3 implementation with clipped double Q-learning and delayed policy updates."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cfg: TD3Config,
        device: torch.device,
    ):
        self.cfg = cfg
        self.device = device
        self.action_dim = action_dim

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

        self.total_updates = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Run actor and optionally add exploration noise for behavior policy."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_tensor).squeeze(0).cpu().numpy()

        if add_noise:
            noise = np.random.normal(0.0, self.cfg.exploration_noise, size=self.action_dim).astype(np.float32)
            action = action + noise

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def train_step(self, replay_buffer: ReplayBuffer) -> Optional[Dict[str, float]]:
        """Perform one TD3 update step if enough samples are available."""
        if len(replay_buffer) < self.cfg.batch_size:
            return None

        self.total_updates += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(self.cfg.batch_size)

        with torch.no_grad():
            # Target policy smoothing regularizes critic targets.
            noise = torch.randn_like(actions) * self.cfg.target_policy_noise
            noise = torch.clamp(noise, -self.cfg.target_noise_clip, self.cfg.target_noise_clip)

            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics: Dict[str, float] = {"critic_loss": float(critic_loss.item())}

        if self.total_updates % self.cfg.policy_delay == 0:
            # Actor maximizes Q-value estimate from first critic.
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1_forward(states, actor_actions).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor, self.cfg.tau)
            self._soft_update(self.critic_target, self.critic, self.cfg.tau)

            metrics["actor_loss"] = float(actor_loss.item())

        return metrics

    @staticmethod
    def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_updates": self.total_updates,
            "td3_config": asdict(self.cfg),
        }
        torch.save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.total_updates = int(payload.get("total_updates", 0))
