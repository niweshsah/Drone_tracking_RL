import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ARCHITECTURE MATCHING UPDATED ENV ---
    state_dim: int = 6       # UPDATED: [x, y, w, h, vx, vy]
    action_dim: int = 2      # [delta_vx, delta_vy]
    hidden_dim: int = 256    # UPDATED: Increased capacity for complex 6D state
    
    # --- PHYSICAL & CONTROL LIMITS ---
    max_action: float = 5.0          # Absolute max velocity of drone
    max_residual_vel: float = 2.0    # ADDED: Max velocity correction RL can apply
    max_accel: float = 10.0          # Physics acceleration limit
    target_vel_noise: float = 0.2    # ADDED: Noise injected into feedforward
    
    # --- TD3 HYPERPARAMETERS ---
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005 
    batch_size: int = 64 
    buffer_size: int = 1_000_000
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # --- EXPLORATION SCHEDULE ---
    random_episodes: int = 50   
    noise_episodes: int = 1450    
    update_interval: int = 50     
    explore_noise: float = 0.15   
    explore_noise_decay: float = 0.998 

    # --- EPISODE LIMITS ---
    total_episodes: int = 2000
    max_episode_steps: int = 500 
    control_period: float = 0.05  # UPDATED: Matched env.py control frequency

    # --- REWARD WEIGHTS ---
    reward_weights: dict = None   # Can be initialized dynamically if needed

    # --- LOGGING ---
    log_dir: str = "runs/vtd3"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_episodes: int = 25

    def __post_init__(self):
        if self.reward_weights is None:
            self.reward_weights = {
                'align': 2.0, 'scale': -1.5, 'smooth': -0.2, 
                'energy': -0.05, 'boundary': -1.0, 'crash': -50.0
            }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_csv_row(csv_path: str, row: Dict[str, float], header: Optional[Iterable[str]] = None) -> None:
    ensure_dir(os.path.dirname(csv_path) or ".")
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(header) if header else list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def moving_average(x: np.ndarray, window: int = 20) -> np.ndarray:
    if len(x) == 0:
        return x
    window = max(1, min(window, len(x)))
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def compute_tracking_metrics(
    drone_xy: np.ndarray,
    target_xy: np.ndarray,
    drone_vel_xy: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """Calculates tracking accuracy and control smoothness metrics."""
    err = drone_xy - target_xy
    x_err = float(np.mean(np.abs(err[:, 0])))
    y_err = float(np.mean(np.abs(err[:, 1])))

    if len(drone_vel_xy) < 3:
        jitter = 0.0
        jerk_rms = 0.0
    else:
        dv = np.diff(drone_vel_xy, axis=0)
        jitter = float(np.mean(np.linalg.norm(dv, axis=1)))

        d2v = np.diff(drone_vel_xy, n=2, axis=0)
        jerk = d2v / (dt ** 2)
        jerk_rms = float(np.sqrt(np.mean(np.sum(jerk ** 2, axis=1))))

    return {
        "x_tracking_error": x_err,
        "y_tracking_error": y_err,
        "velocity_jitter": jitter,
        "jerk_rms": jerk_rms,
    }


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def exponential_decay_schedule(start: float, decay: float, step: int) -> float:
    """Calculates exponential decay (formerly named linear_schedule)."""
    return start * (decay ** step)