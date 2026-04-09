from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class VisionConfig:
    img_width: int = 640
    img_height: int = 640
    fov_deg: float = 90.0
    target_real_width: float = 1.8
    target_real_height: float = 1.6
    detection_noise_std_x: float = 0.003
    detection_noise_std_area: float = 0.002
    dropout_prob: float = 0.0


class VisionBBoxEstimator:
    """
    Headless, analytic approximation of YOLOv8 + BoT-SORT output.

    It computes a normalized bbox center and area from relative geometry, then applies
    optional noise/dropout to emulate real perception uncertainty.
    """

    def __init__(self, cfg: VisionConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self._last_bbox: Optional[Tuple[float, float]] = None

        fov = np.deg2rad(cfg.fov_deg)
        self.fx = (cfg.img_width / 2.0) / np.tan(fov / 2.0)

    def reset(self) -> None:
        self._last_bbox = None

    def project_bbox(
        self,
        drone_pos: np.ndarray,
        drone_yaw: float,
        target_pos: np.ndarray,
        occluded: bool = False,
    ) -> Dict[str, float]:
        rel_world = target_pos - drone_pos

        # Rotate world relative vector to camera frame (camera looks along +x body axis).
        c = np.cos(-drone_yaw)
        s = np.sin(-drone_yaw)
        x_cam = c * rel_world[0] - s * rel_world[1]
        y_cam = s * rel_world[0] + c * rel_world[1]
        # Target behind camera or too close: treat as loss and fallback to tracker prediction.
        if x_cam <= 0.2:
            return self._fallback_bbox(lost=True)

        u_px = self.fx * (y_cam / x_cam) + (self.cfg.img_width / 2.0)
        x_center = np.clip(u_px / self.cfg.img_width, 0.0, 1.0)

        bbox_w_px = self.fx * self.cfg.target_real_width / x_cam
        bbox_h_px = self.fx * self.cfg.target_real_height / x_cam
        area_norm = np.clip((bbox_w_px * bbox_h_px) / (self.cfg.img_width * self.cfg.img_height), 0.0, 1.0)

        if occluded or self.rng.random() < self.cfg.dropout_prob:
            return self._fallback_bbox(lost=True)

        x_center += self.rng.normal(0.0, self.cfg.detection_noise_std_x)
        area_norm += self.rng.normal(0.0, self.cfg.detection_noise_std_area)
        x_center = float(np.clip(x_center, 0.0, 1.0))
        area_norm = float(np.clip(area_norm, 1e-6, 1.0))

        self._last_bbox = (x_center, area_norm)
        return {
            "x_center": x_center,
            "area": area_norm,
            "detected": 1.0,
            "depth_proxy": float(1.0 / np.sqrt(area_norm + 1e-8)),
        }

    def _fallback_bbox(self, lost: bool) -> Dict[str, float]:
        # Mimics short-term tracker persistence under occlusion.
        if self._last_bbox is None:
            x_center = 0.5
            area_norm = 1e-3
        else:
            x_center, area_norm = self._last_bbox
            x_center = float(np.clip(x_center + self.rng.normal(0.0, 0.005), 0.0, 1.0))
            area_norm = float(np.clip(area_norm * 0.995, 1e-6, 1.0))

        self._last_bbox = (x_center, area_norm)
        return {
            "x_center": x_center,
            "area": area_norm,
            "detected": 0.0 if lost else 1.0,
            "depth_proxy": float(1.0 / np.sqrt(area_norm + 1e-8)),
        }


def vision_state_from_bbox(x_box: float, s_box: float, x_des: float, s_des: float) -> np.ndarray:
    s1 = (x_box - x_des) / max(x_des, 1e-8)
    s2 = (s_box - s_des) / max(s_des, 1e-8)
    return np.asarray([s1, s2], dtype=np.float32)
