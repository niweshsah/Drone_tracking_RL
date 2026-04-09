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

    # Paper-specified architecture (Table 2) [cite: 626]
    state_dim: int = 2
    action_dim: int = 2
    max_action: float = 6.0 
    hidden_dim: int = 64
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    gamma: float = 0.99
    tau: float = 0.005 # Note: Paper mentions soft update but uses standard TD3 tau
    batch_size: int = 64 
    buffer_size: int = 1_000_000
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # Exploration schedule (Aligned with Figure 10 & Algorithm 1) 
    random_episodes: int = 1000   # Phase 1: Episodes 0-1000 [cite: 665]
    noise_episodes: int = 500    # Phase 2: Episodes 1000-1500 [cite: 665]
    # Total episodes is 2000; Phase 3 (Pure Policy) is 1500-2000 [cite: 665]
    
    update_interval: int = 50     # Update every 50 environment steps [cite: 626]
    explore_noise: float = 0.15   # Initial Gaussian noise magnitude [cite: 626]
    explore_noise_decay: float = 0.998 # Decay rate per episode [cite: 626]

    total_episodes: int = 2000
    max_episode_steps: int = 2000 # Paper uses MAX_STEPS 2000 total, but 300/ep is standard for tracking
    control_period: float = 0.3  # 0.3s per step [cite: 626]

    # Reward weights (Implementation specific based on Equation 9) [cite: 370]
    w1: float = 1.0  # Weight for Rs (Distance)
    w2: float = 0.15 # Weight for Rspeed
    w3: float = 0.15 # Weight for Rstability
    
    x_des: float = 0.5               # Desired x_center in normalized image coordinates
    s_des: float = 0.06              # Desired target area/scale in camera view

    log_dir: str = "runs/vtd3"
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval_episodes: int = 25


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


def linear_schedule(start: float, decay: float, step: int) -> float:
    return start * (decay ** step)
