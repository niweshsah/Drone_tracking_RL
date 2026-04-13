from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple Replay Buffer for TD3
class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 1_000_000):
        self.max_size = max_size # Maximum number of transitions to store
        self.ptr = 0  # Pointer to the current index for adding new transitions
        self.size = 0 # Current size of the buffer (number of transitions stored)

        self.state = np.zeros((max_size, state_dim), dtype=np.float32) # Pre-allocate memory for states
        self.action = np.zeros((max_size, action_dim), dtype=np.float32) # Pre-allocate memory for actions
        self.reward = np.zeros((max_size, 1), dtype=np.float32) # Pre-allocate memory for rewards
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32) # Pre-allocate memory for next states
        self.done = np.zeros((max_size, 1), dtype=np.float32) # Pre-allocate memory for done flags (1 if episode ended, else 0)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # return (state, action, reward, next_state, done) as tensors on the specified device
    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
        idx = np.random.randint(0, self.size, size=batch_size) # Randomly sample indices for the batch
        s = torch.as_tensor(self.state[idx], dtype=torch.float32, device=device)
        a = torch.as_tensor(self.action[idx], dtype=torch.float32, device=device)
        r = torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device)
        ns = torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device)
        d = torch.as_tensor(self.done[idx], dtype=torch.float32, device=device)
        return s, a, r, ns, d

# TD3 Agent Implementation
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dim: int = 64):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim) # First hidden layer
        self.l2 = nn.Linear(hidden_dim, hidden_dim) # Second hidden layer
        self.l3 = nn.Linear(hidden_dim, action_dim) # Output layer for action prediction
        self.max_action = max_action # Maximum action value for scaling the output (since actions are typically in a certain range)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_dim = state_dim + action_dim # The critic takes both state and action as input to evaluate the Q-value

        self.q1_l1 = nn.Linear(in_dim, hidden_dim) # First hidden layer for Q1
        self.q1_l2 = nn.Linear(hidden_dim, hidden_dim) # Second hidden layer for Q1
        self.q1_l3 = nn.Linear(hidden_dim, 1) # Output layer for Q1 value prediction

        self.q2_l1 = nn.Linear(in_dim, hidden_dim) # First hidden layer for Q2 (TD3 uses two critics to mitigate overestimation bias)
        self.q2_l2 = nn.Linear(hidden_dim, hidden_dim) # Second hidden layer for Q2
        self.q2_l3 = nn.Linear(hidden_dim, 1) # Output layer for Q2 value prediction

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=1) 

        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        q1 = self.q1_l3(q1)

        q2 = F.relu(self.q2_l1(sa))
        q2 = F.relu(self.q2_l2(q2))
        q2 = self.q2_l3(q2)
        return q1, q2

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor: # This function is used to compute only the Q1 value, which is needed for the actor loss calculation in TD3
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_l1(sa))
        q1 = F.relu(self.q1_l2(q1))
        return self.q1_l3(q1)


@dataclass
class TD3Config: # Configuration dataclass for TD3 hyperparameters and dimensions
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


class TD3Agent: # there are 6 neural networks in total: actor, critic, and their respective target networks. The target networks are used to stabilize training by providing a slowly changing target for the critic updates.
    def __init__(self, cfg: TD3Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Initialize actor and critic networks along with their target networks
        self.actor = Actor(cfg.state_dim, cfg.action_dim, cfg.max_action, cfg.hidden_dim).to(self.device)
        self.actor_target = Actor(cfg.state_dim, cfg.action_dim, cfg.max_action, cfg.hidden_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict()) # Initialize target actor with the same weights as the main actor

        self.critic = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device) # Main critic network
        self.critic_target = Critic(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device) # Target critic network
        self.critic_target.load_state_dict(self.critic.state_dict()) # Initialize target critic with the same weights as the main critic

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr) # Optimizer for the actor network
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr) # Optimizer for the critic network

        self.total_it = 0

    @torch.no_grad() # This decorator indicates that the following function should not compute gradients, which is important for efficiency during action selection
    def select_action(self, state: np.ndarray) -> np.ndarray: # 
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(state_t).cpu().numpy()[0]

    # returns a dictionary of losses for logging purposes
    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int) -> Dict[str, float]: 
        self.total_it += 1

        state, action, reward, next_state, done = replay_buffer.sample(batch_size, self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.cfg.policy_noise).clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.cfg.max_action, self.cfg.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action) # Get the target Q-values from the target critic network for the next state and the action predicted by the target actor (with added noise for exploration)
            
            target_q = torch.min(target_q1, target_q2)
            
            y = reward + (1.0 - done) * self.cfg.gamma * target_q # Bellman equation


        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y) # Compute the critic loss as the mean squared error between the current Q-values and the target Q-values for both critics


        self.critic_opt.zero_grad(set_to_none=True) # Clear the gradients of the critic optimizer before backpropagation
        critic_loss.backward() # Backpropagate the critic loss to compute gradients for the critic network parameters
        self.critic_opt.step() # Update the critic network parameters using the optimizer


        actor_loss = torch.tensor(0.0, device=self.device) # create a placholder for actor loss
        
        
        
        # Delayed actor updates
        if self.total_it % self.cfg.policy_delay == 0: # Delayed policy updates: The actor is updated less frequently than the critic
            actor_loss = -self.critic.q1(state, self.actor(state)).mean() # compute the actor loss as the negative mean Q-value predicted by the critic for the actions output by the actor (we want to maximize this Q-value, hence the negative sign)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.critic_target, self.critic)
            self._soft_update(self.actor_target, self.actor)



        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }





    def _soft_update(self, target: nn.Module, source: nn.Module): # update the target networks
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
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic.load_state_dict(payload["critic"], strict=strict)
        self.actor_target.load_state_dict(payload["actor_target"], strict=strict)
        self.critic_target.load_state_dict(payload["critic_target"], strict=strict)
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.total_it = int(payload.get("total_it", 0))
