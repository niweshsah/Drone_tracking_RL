"""Reward model for vision-based UAV target tracking in a POMDP setup."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from config import RewardConfig


@dataclass
class RewardTerms:
    """Container returned for debugging each reward component."""

    align: float
    scale: float
    smooth: float
    boundary: float
    crash: float


class DenseTrackingReward:
    """Computes dense multi-objective reward based on tracking quality and control.

    Expected frame format:
        [x_norm, y_norm, w_norm, h_norm, v_f_norm, v_l_norm]
    """

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def compute(
        self,
        frame: np.ndarray,
        action: np.ndarray,
        prev_action: np.ndarray,
        target_lost: bool = False,
        out_of_bounds: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total reward and per-component terms.

        Args:
            frame: Current normalized 6D observation frame.
            action: Current action a_t in [-1, 1]^2.
            prev_action: Previous action a_{t-1} in [-1, 1]^2.
            target_lost: Whether the target is no longer observable.
            out_of_bounds: Whether platform/target exceeds operational boundary.
        """
        x_norm, y_norm, w_norm, h_norm, _, _ = frame

        # Alignment term: high when target center is close to image center.
        radial_error = float(np.sqrt(x_norm**2 + y_norm**2))
        r_align = self.cfg.c1 * np.exp(-self.cfg.alpha * radial_error)

        # Scale term: high when target occupies an expected area in the image.
        area = float(w_norm * h_norm)
        r_scale = -self.cfg.c2 * abs(area - self.cfg.a_optimal)

        # Smoothness term: penalizes abrupt control jumps (bang-bang behavior).
        action_delta = action - prev_action
        r_smooth = -self.cfg.c3 * float(np.dot(action_delta, action_delta))

        # Boundary term: zero in safe region, linearly negative outside it.
        r_boundary = 0.0
        if radial_error > self.cfg.r_safe:
            r_boundary = -self.cfg.c4 * (radial_error - self.cfg.r_safe)

        # Crash/terminal term: heavy penalty for losing target or mission failure.
        r_crash = self.cfg.crash_penalty if (target_lost or out_of_bounds) else 0.0

        total = r_align + r_scale + r_smooth + r_boundary + r_crash
        terms = RewardTerms(
            align=float(r_align),
            scale=float(r_scale),
            smooth=float(r_smooth),
            boundary=float(r_boundary),
            crash=float(r_crash),
        )
        return float(total), terms.__dict__
