
import numpy as np
from typing import Tuple


class TargetTrajectory:
    def __init__(self,  start_xy: Tuple[float, float], scale: float = 50.0, mode: str = "triangular"):
        self.mode = mode
        self.segment_duration = 16.5
        self.pause_duration = 2.0
        
        start_pos = np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float32)
        L = scale 

        if mode == "triangular":
            offsets = [[0, 0, 0], [L, L, 0], [2*L, 0, 0], [0, 0, 0]]
        elif mode == "square":
            offsets = [[0, 0, 0], [0, L, 0], [L, L, 0], [L, 0, 0], [0, 0, 0]]
        elif mode == "sawtooth":
            offsets = [[0, 0, 0], [0.6*L, 0.8*L, 0], [0.8*L, -0.1*L, 0], [1.4*L, 0.9*L, 0], [1.6*L, -0.2*L, 0], [2.2*L, 0.8*L, 0]]
        elif mode == "square_wave":
            offsets = [[0, 0, 0], [0.6*L, 0, 0], [0.6*L, 0.8*L, 0], [1.4*L, 0.8*L, 0], [1.4*L, 0, 0], [2.2*L, 0, 0], [2.2*L, 0.8*L, 0], [3.0*L, 0.8*L, 0]]
        else:
            raise ValueError(f"Unknown trajectory mode: {mode}")

        self.points = [start_pos + np.array(off, dtype=np.float32) for off in offsets]
        self.n_segments = len(self.points) - 1

    def duration(self) -> float:
        return self.n_segments * (self.segment_duration + self.pause_duration)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        cycle = self.duration()
        t_mod = t % cycle

        segment_full = self.segment_duration + self.pause_duration
        seg_idx = min(int(np.floor(t_mod / segment_full)), self.n_segments - 1)

        t_local = t_mod - seg_idx * segment_full
        p0, p1 = self.points[seg_idx], self.points[seg_idx + 1]

        if t_local >= self.segment_duration:
            return p1.copy(), np.zeros(3, dtype=np.float32)

        T = self.segment_duration
        ta, tc, td = 0.25 * T, 0.5 * T, 0.25 * T  
        d = p1 - p0
        dist = np.linalg.norm(d)
        direction = d / (dist + 1e-8)

        vmax = dist / (0.5 * ta + tc + 0.5 * td + 1e-8)
        a = vmax / (ta + 1e-8)

        if t_local < ta:
            s, v = 0.5 * a * t_local**2, a * t_local
        elif t_local < (ta + tc):
            dt2 = t_local - ta
            s, v = 0.5 * a * ta**2 + vmax * dt2, vmax
        else:
            dt3 = t_local - ta - tc
            s = 0.5 * a * ta**2 + vmax * tc + (vmax * dt3 - 0.5 * a * dt3**2)
            v = max(0.0, vmax - a * dt3)

        return (p0 + direction * min(s, dist)), (direction * v)

