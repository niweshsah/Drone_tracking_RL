"""Gymnasium wrapper for vision-based UAV target tracking POMDP formatting."""

from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import EnvConfig
from reward import DenseTrackingReward


class VisionTrackingPOMDPWrapper(gym.Wrapper):
    """Wraps a base Gym env and emits stacked temporal state vectors.

    Base frame format (6D):
        [x_norm, y_norm, w_norm, h_norm, v_f_norm, v_l_norm]

    Output state format (24D):
        concat(last k=4 frames), oldest -> newest
    """

    def __init__(
        self,
        env: gym.Env,
        env_cfg: EnvConfig,
        reward_model: DenseTrackingReward,
    ):
        super().__init__(env)
        self.cfg = env_cfg
        self.reward_model = reward_model

        self.history: Deque[np.ndarray] = deque(maxlen=self.cfg.history_len)
        self.prev_action = np.zeros(self.cfg.action_dim, dtype=np.float32)

        frame_low = np.array([-1.0, -1.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32)
        frame_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.tile(frame_low, self.cfg.history_len),
            high=np.tile(frame_high, self.cfg.history_len),
            shape=(self.cfg.state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_dim,),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        frame = self._extract_frame(obs, info)

        self.history.clear()
        for _ in range(self.cfg.history_len):
            self.history.append(frame.copy())

        self.prev_action = np.zeros(self.cfg.action_dim, dtype=np.float32)
        stacked_state = self._stack_history()
        return stacked_state, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        obs, _, terminated, truncated, info = self.env.step(action)
        frame = self._extract_frame(obs, info)

        target_lost = bool(info.get("target_lost", False))
        out_of_bounds = bool(info.get("out_of_bounds", False))

        reward, reward_terms = self.reward_model.compute(
            frame=frame,
            action=action,
            prev_action=self.prev_action,
            target_lost=target_lost,
            out_of_bounds=out_of_bounds,
        )

        self.prev_action = action.copy()
        self.history.append(frame)

        # Failure signals from tracking logic should end the episode immediately.
        terminated = bool(terminated or target_lost or out_of_bounds)
        info = dict(info)
        info["reward_terms"] = reward_terms

        return self._stack_history(), reward, terminated, truncated, info

    def _stack_history(self) -> np.ndarray:
        return np.concatenate(list(self.history), axis=0).astype(np.float32)

    def _extract_frame(self, obs: Any, info: Dict[str, Any]) -> np.ndarray:
        """Extract and normalize the 6D frame from base-env observations.

        This intentionally accepts flexible mock inputs so the RL stack can be
        tested without committing to a single CV/physics integration format.
        """
        if isinstance(obs, np.ndarray) and obs.shape == (self.cfg.frame_dim,):
            frame = obs.astype(np.float32)
        elif isinstance(obs, (list, tuple)) and len(obs) == self.cfg.frame_dim:
            frame = np.asarray(obs, dtype=np.float32)
        elif isinstance(obs, dict):
            # Preferred dict keys when integrating with detector + ego-state.
            x = float(obs.get("x_norm", 0.0))
            y = float(obs.get("y_norm", 0.0))
            w = float(obs.get("w_norm", 0.0))
            h = float(obs.get("h_norm", 0.0))
            vf = float(obs.get("v_f_norm", 0.0))
            vl = float(obs.get("v_l_norm", 0.0))
            frame = np.array([x, y, w, h, vf, vl], dtype=np.float32)
        else:
            # Fallback for unexpected formats to keep training loop robust.
            frame = np.zeros(self.cfg.frame_dim, dtype=np.float32)

        # Enforce specified normalization bounds for each channel.
        frame[0] = np.clip(frame[0], -1.0, 1.0)
        frame[1] = np.clip(frame[1], -1.0, 1.0)
        frame[2] = np.clip(frame[2], 0.0, 1.0)
        frame[3] = np.clip(frame[3], 0.0, 1.0)
        frame[4] = np.clip(frame[4], -1.0, 1.0)
        frame[5] = np.clip(frame[5], -1.0, 1.0)

        return frame
