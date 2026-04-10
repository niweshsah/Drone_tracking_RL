from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        s = torch.as_tensor(self.state[idx], dtype=torch.float32, device=device)
        a = torch.as_tensor(self.action[idx], dtype=torch.float32, device=device)
        r = torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device)
        ns = torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device)
        d = torch.as_tensor(self.done[idx], dtype=torch.float32, device=device)
        return s, a, r, ns, d


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = state_dim + action_dim

        self.q1_l1 = nn.Linear(in_dim, hidden_dim)
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_l3 = nn.Linear(hidden_dim, 1)

        self.q2_l1 = nn.Linear(in_dim, hidden_dim)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


@dataclass
class TD3Config:
    state_dim: int
    action_dim: int
    max_action: float
    hidden_dim: int = 64
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    device: str = "cpu"


class TD3Agent:
    def __init__(self, cfg: TD3Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.max_action, cfg.hidden_dim).to(self.device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim, cfg.max_action, cfg.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.total_it = 0

    @torch.no_grad()
    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(state_t).cpu().numpy()[0]

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]:
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.cfg.max_action, self.cfg.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            y = reward + (1.0 - done) * self.cfg.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.cfg.policy_delay == 0:
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.actor_target, self.actor)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module):
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.copy_(self.cfg.tau * p.data + (1.0 - self.cfg.tau) * p_t.data)

    def save(self, path: str) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_it": self.total_it,
            "cfg": self.cfg.__dict__,
        }
        torch.save(payload, path)

    def load(self, path: str, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic.load_state_dict(payload["critic"], strict=strict)
        self.actor_target.load_state_dict(payload["actor_target"], strict=strict)
        self.critic_target.load_state_dict(payload["critic_target"], strict=strict)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.total_it = int(payload.get("total_it", 0))
