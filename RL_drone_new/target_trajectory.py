
# import numpy as np
# from typing import Tuple


# class TargetTrajectory:
#     def __init__(self,  start_xy: Tuple[float, float], scale: float = 50.0, mode: str = "triangular"):
#         self.mode = mode
#         self.segment_duration = 16.5
#         self.pause_duration = 2.0
        
#         start_pos = np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float32)
#         L = scale 

#         if mode == "triangular":
#             offsets = [[0, 0, 0], [L, L, 0], [2*L, 0, 0], [0, 0, 0]]
#         elif mode == "square":
#             offsets = [[0, 0, 0], [0, L, 0], [L, L, 0], [L, 0, 0], [0, 0, 0]]
#         elif mode == "sawtooth":
#             offsets = [[0, 0, 0], [0.6*L, 0.8*L, 0], [0.8*L, -0.1*L, 0], [1.4*L, 0.9*L, 0], [1.6*L, -0.2*L, 0], [2.2*L, 0.8*L, 0]]
#         elif mode == "square_wave":
#             offsets = [[0, 0, 0], [0.6*L, 0, 0], [0.6*L, 0.8*L, 0], [1.4*L, 0.8*L, 0], [1.4*L, 0, 0], [2.2*L, 0, 0], [2.2*L, 0.8*L, 0], [3.0*L, 0.8*L, 0]]
#         else:
#             raise ValueError(f"Unknown trajectory mode: {mode}")

#         self.points = [start_pos + np.array(off, dtype=np.float32) for off in offsets]
#         self.n_segments = len(self.points) - 1

#     def duration(self) -> float:
#         return self.n_segments * (self.segment_duration + self.pause_duration)

#     def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
#         cycle = self.duration()
#         t_mod = t % cycle

#         segment_full = self.segment_duration + self.pause_duration
#         seg_idx = min(int(np.floor(t_mod / segment_full)), self.n_segments - 1)

#         t_local = t_mod - seg_idx * segment_full
#         p0, p1 = self.points[seg_idx], self.points[seg_idx + 1]

#         if t_local >= self.segment_duration:
#             return p1.copy(), np.zeros(3, dtype=np.float32)

#         T = self.segment_duration
#         ta, tc, td = 0.25 * T, 0.5 * T, 0.25 * T  
#         d = p1 - p0
#         dist = np.linalg.norm(d)
#         direction = d / (dist + 1e-8)

#         vmax = dist / (0.5 * ta + tc + 0.5 * td + 1e-8)
#         a = vmax / (ta + 1e-8)

#         if t_local < ta:
#             s, v = 0.5 * a * t_local**2, a * t_local
#         elif t_local < (ta + tc):
#             dt2 = t_local - ta
#             s, v = 0.5 * a * ta**2 + vmax * dt2, vmax
#         else:
#             dt3 = t_local - ta - tc
#             s = 0.5 * a * ta**2 + vmax * tc + (vmax * dt3 - 0.5 * a * dt3**2)
#             v = max(0.0, vmax - a * dt3)

#         return (p0 + direction * min(s, dist)), (direction * v)






import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple

class TargetTrajectory:
    def __init__(self, start_xy: Tuple[float, float], scale: float = 50.0, mode: str = "triangular", seed: int = None):
        self.mode = mode
        self.segment_duration = 16.5
        self.pause_duration = 2.0
        
        start_pos = np.array([start_xy[0], start_xy[1], 0.0], dtype=np.float32)
        L = scale 

        # --- STATIC SHAPES ---
        if mode in ["triangular", "square", "sawtooth", "square_wave"]:
            if mode == "triangular":
                offsets = [[0, 0, 0], [L, L, 0], [2*L, 0, 0], [0, 0, 0]]
            elif mode == "square":
                offsets = [[0, 0, 0], [0, L, 0], [L, L, 0], [L, 0, 0], [0, 0, 0]]
            elif mode == "sawtooth":
                offsets = [[0, 0, 0], [0.6*L, 0.8*L, 0], [0.8*L, -0.1*L, 0], [1.4*L, 0.9*L, 0], [1.6*L, -0.2*L, 0], [2.2*L, 0.8*L, 0]]
            elif mode == "square_wave":
                offsets = [[0, 0, 0], [0.6*L, 0, 0], [0.6*L, 0.8*L, 0], [1.4*L, 0.8*L, 0], [1.4*L, 0, 0], [2.2*L, 0, 0], [2.2*L, 0.8*L, 0], [3.0*L, 0.8*L, 0]]
            
            self.points = [start_pos + np.array(off, dtype=np.float32) for off in offsets]
            self.n_segments = len(self.points) - 1

        # --- RANDOM SPLINE CURRICULUM ---
        elif mode.startswith("spline_"):
            # Set Difficulty Parameters
            if mode == "spline_easy":
                self.v_limit = 2.0   # Slow speed
                self.a_limit = 1.0   # Gentle acceleration
                n_waypoints = 4      # Long, sweeping, predictable curves
            elif mode == "spline_medium":
                self.v_limit = 5.0   # Standard speed
                self.a_limit = 3.0   # Moderate braking/turning
                n_waypoints = 8      # Normal erratic behavior
            elif mode == "spline_hard":
                self.v_limit = 10.0  # Very fast (approaching drone max speed)
                self.a_limit = 8.0   # Highly aggressive maneuvers
                n_waypoints = 15     # Tight, unpredictable zig-zags
            else:
                raise ValueError(f"Unknown spline mode: {mode}")
            
            rng = np.random.default_rng(seed)
            
            # 1. Generate random waypoints within bounds [-L, L]
            points = rng.uniform(-L, L, size=(n_waypoints, 3))
            points[:, 2] = 0.0 # Keep it on the ground plane
            
            # Force start point and close the loop for continuous cyclic flying
            points[0] = start_pos
            points[-1] = start_pos
            self.points = points
            
            # 2. Random time allocation based on chordal distance
            dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
            random_factors = rng.uniform(0.5, 1.5, size=len(dists))
            segment_times = dists * random_factors
            
            t_raw = np.zeros(len(points))
            t_raw[1:] = np.cumsum(segment_times)
            t_raw /= t_raw[-1] # Normalize time to [0, 1]
            
            # 3. Fit periodic Cubic Spline
            self.spline = CubicSpline(t_raw, points, bc_type='periodic')
            self.spline_v = self.spline.derivative(1)
            self.spline_a = self.spline.derivative(2)
            
            # 4. Enforce max velocity and max acceleration limits via Time Scaling
            t_fine = np.linspace(0, 1, 1000)
            v_raw_max = np.max(np.linalg.norm(self.spline_v(t_fine), axis=1))
            a_raw_max = np.max(np.linalg.norm(self.spline_a(t_fine), axis=1))
            
            T_v = v_raw_max / self.v_limit
            T_a = np.sqrt(a_raw_max / self.a_limit)
            
            self.T_total = max(T_v, T_a)
            
        else:
            raise ValueError(f"Unknown trajectory mode: {mode}")
        

    def duration(self) -> float:
        if self.mode.startswith("spline_"):
            return float(self.T_total)
        return self.n_segments * (self.segment_duration + self.pause_duration)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        cycle = self.duration()
        t_mod = t % cycle

        if self.mode.startswith("spline_"):
            tau = t_mod / self.T_total
            pos = self.spline(tau)
            vel = self.spline_v(tau) / self.T_total
            return pos.astype(np.float32), vel.astype(np.float32)

        # Static shape logic
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