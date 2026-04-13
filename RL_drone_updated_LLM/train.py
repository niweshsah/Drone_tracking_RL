"""End-to-end TD3 training loop with a mock Gymnasium UAV tracking environment.

The mock environment stands in for gym-pybullet-drones + YOLO processing,
so the code focuses on RL logic, state processing, and TD3 updates.
"""

from __future__ import annotations

import os
import random
from collections import deque
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from config import ProjectConfig
from env_wrapper import VisionTrackingPOMDPWrapper
from replay_buffer import ReplayBuffer
from reward import DenseTrackingReward
from td3_agent import TD3Agent


class MockDroneTrackingEnv(gym.Env):
    """Mock base environment with continuous action and 6D normalized frame output.

    Observation (single frame):
        [x_norm, y_norm, w_norm, h_norm, v_f_norm, v_l_norm]
    Action:
        [v_x_cmd, v_y_cmd] in [-1, 1]

    Notes:
    - This environment intentionally approximates dynamics with simple equations.
    - It provides failure flags via info to mimic target loss / out-of-bounds events.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 300):
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            shape=(6,),
            dtype=np.float32,
        )

        self.state = np.zeros(6, dtype=np.float32)
        self.rng = np.random.default_rng()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0

        # Start from mildly off-center target and plausible box scale.
        x = self.rng.uniform(-0.4, 0.4)
        y = self.rng.uniform(-0.4, 0.4)
        w = self.rng.uniform(0.2, 0.45)
        h = self.rng.uniform(0.2, 0.45)
        vf = 0.0
        vl = 0.0
        self.state = np.array([x, y, w, h, vf, vl], dtype=np.float32)

        return self.state.copy(), {}

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        x, y, w, h, vf, vl = self.state

        # Controlled center-offset dynamics: command counters target image offset.
        process_noise = self.rng.normal(0.0, 0.02, size=2).astype(np.float32)
        x = x - 0.10 * action[0] + process_noise[0]
        y = y - 0.10 * action[1] + process_noise[1]

        # Ego-velocity channels are tied to commanded velocities with slight lag.
        vf = np.clip(0.85 * vf + 0.15 * action[0], -1.0, 1.0)
        vl = np.clip(0.85 * vl + 0.15 * action[1], -1.0, 1.0)

        # Box width/height drift around a plausible area with small noise.
        w = np.clip(w + self.rng.normal(0.0, 0.01), 0.05, 0.95)
        h = np.clip(h + self.rng.normal(0.0, 0.01), 0.05, 0.95)

        self.state = np.array([x, y, w, h, vf, vl], dtype=np.float32)

        out_of_bounds = bool(abs(x) > 1.1 or abs(y) > 1.1)
        target_lost = bool(abs(x) > 1.0 or abs(y) > 1.0)

        terminated = out_of_bounds
        truncated = self.step_count >= self.max_steps

        # Base reward is ignored by the POMDP wrapper (it computes dense reward).
        reward = 0.0

        info = {
            "target_lost": target_lost,
            "out_of_bounds": out_of_bounds,
        }
        return self.state.copy(), reward, terminated, truncated, info


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: Optional[ProjectConfig] = None) -> None:
    cfg = config or ProjectConfig()
    set_seed(cfg.train.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_env = MockDroneTrackingEnv(max_steps=cfg.env.max_episode_steps)
    reward_model = DenseTrackingReward(cfg.reward)
    env = VisionTrackingPOMDPWrapper(base_env, cfg.env, reward_model)

    agent = TD3Agent(
        state_dim=cfg.env.state_dim,
        action_dim=cfg.env.action_dim,
        cfg=cfg.td3,
        device=device,
    )

    replay_buffer = ReplayBuffer(
        state_dim=cfg.env.state_dim,
        action_dim=cfg.env.action_dim,
        capacity=cfg.td3.buffer_size,
        device=device,
    )

    global_step = 0
    reward_window = deque(maxlen=20)

    os.makedirs(cfg.train.checkpoint_dir, exist_ok=True)

    for episode in range(1, cfg.train.episodes + 1):
        state, _ = env.reset(seed=cfg.train.seed + episode)
        episode_reward = 0.0

        for _ in range(cfg.train.max_steps_per_episode):
            if global_step < cfg.train.start_timesteps:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)

            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            global_step += 1

            if global_step >= cfg.train.update_after:
                for _ in range(cfg.train.updates_per_step):
                    agent.train_step(replay_buffer)

            if done:
                break

        reward_window.append(episode_reward)
        avg_reward = float(np.mean(reward_window))
        print(
            f"Episode {episode:04d} | "
            f"EpReward: {episode_reward:8.3f} | "
            f"Avg20: {avg_reward:8.3f} | "
            f"Buffer: {len(replay_buffer):7d}"
        )

        if episode % cfg.train.checkpoint_every == 0:
            ckpt_path = os.path.join(cfg.train.checkpoint_dir, f"td3_ep_{episode:04d}.pt")
            agent.save(ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    env.close()


if __name__ == "__main__":
    train()
