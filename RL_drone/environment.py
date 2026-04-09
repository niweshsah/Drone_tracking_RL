import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Note: These imports assume you have 'vision.py' in your path as per your snippet.
try:
    from vision import VisionBBoxEstimator, VisionConfig, vision_state_from_bbox
except ImportError:
    # Fallback placeholders if vision.py is not present locally
    pass



@dataclass # This decorator automatically generates __init__, __repr__, and other methods based on class attributes.
class EnvConfig:
    """
    Hyperparameters for the Simulation and Drone Dynamics.
    Defining these in a dataclass allows for easy tuning and experiment tracking.
    """
    seed: int = 42
    control_period: float = 0.3      # Time (s) between RL agent decisions
    sim_dt: float = 1.0 / 120.0      # High-frequency physics step (8.33ms)
    max_episode_steps: int = 300     # Maximum length of one training episode

    # Environment Bounds and Altitudes
    world_x: float = 220.0           # Boundary of the flying area (meters)
    world_y: float = 220.0
    drone_altitude: float = 4.0      # Drone maintains a fixed flight ceiling
    target_altitude: float = 0.0     # Target moves on the ground

    # Physics Limits
    max_action: float = 6.0          # Max velocity command (m/s)
    max_accel: float = 4.0           # Max allowed acceleration (m/s^2)
    drag: float = 0              # Simple linear air resistance coefficient

    # Visual State Targets (Normalization constants)
    x_des: float = 0.5               # Desired target x-center in camera (0.0 to 1.0)
    s_des: float = 0.06              # Desired target area/scale in camera view

    # Reward Weighting Factors (Balances tracking vs. stability)
    w1: float = 1.0                  # Weight for state error reduction
    w2: float = 0.15                 # Weight for speed incentives
    w3: float = 0.15                 # Weight for hovering stability

    vision: VisionConfig = VisionConfig()





class TargetTrajectory:
    """
    Handles the motion logic for the ground target.
    Calculates where the target should be at any given time 't' based on a 
    predefined geometric path.
    """
    def __init__(self, mode: str):
        self.mode = mode
        self.segment_duration = 16.5   # Time to travel between two waypoints
        self.pause_duration = 2.0      # Time target stays still at waypoints
        self.start = np.array([30.0, 100.0, 0.0], dtype=np.float32)

        # Define waypoints for various difficulty patterns
        if mode == "triangular":
            points = [[30, 100, 0], [80, 150, 0], [130, 100, 0], [30, 100, 0]]
        elif mode == "square":
            points = [[30, 100, 0], [30, 150, 0], [80, 150, 0], [80, 100, 0], [30, 100, 0]]
        elif mode == "sawtooth":
            points = [[30, 100, 0], [60, 140, 0], [70, 95, 0], [100, 145, 0], [110, 90, 0], [140, 140, 0]]
        elif mode == "square_wave":
            points = [[30, 100, 0], [60, 100, 0], [60, 140, 0], [100, 140, 0], [100, 100, 0], [140, 100, 0], [140, 140, 0], [180, 140, 0]]
        else:
            raise ValueError(f"Unknown trajectory mode: {mode}")

        self.points = [np.asarray(pt, dtype=np.float32) for pt in points]
        self.n_segments = len(self.points) - 1

    def duration(self) -> float:
        """Total time for one full loop of the trajectory."""
        return self.n_segments * (self.segment_duration + self.pause_duration)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes (Position, Velocity) at time 't'.
        Uses a trapezoidal velocity profile (acceleration -> constant speed -> deceleration)
        to ensure smooth movement between waypoints.
        """
        cycle = self.duration()
        t_mod = t % cycle

        segment_full = self.segment_duration + self.pause_duration
        seg_idx = min(int(np.floor(t_mod / segment_full)), self.n_segments - 1)

        t_local = t_mod - seg_idx * segment_full
        p0, p1 = self.points[seg_idx], self.points[seg_idx + 1]

        # Handle target pausing at waypoints
        if t_local >= self.segment_duration:
            return p1.copy(), np.zeros(3, dtype=np.float32)

        # Solve for trapezoidal motion parameters
        T = self.segment_duration
        ta, tc, td = 0.25 * T, 0.5 * T, 0.25 * T  # Phase durations
        d = p1 - p0
        dist = np.linalg.norm(d)
        direction = d / (dist + 1e-8)

        # Solve for vmax so the integral of velocity equals the distance
        vmax = dist / (0.5 * ta + tc + 0.5 * td + 1e-8)
        a = vmax / (ta + 1e-8)

        # Compute instantaneous scalar distance 's' and velocity 'v'
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





class DroneTrackingEnv(gym.Env):
    """
    Custom Gymnasium Environment for Visual Drone Tracking.
    The drone (agent) must command XY velocities to keep a moving target 
    in the center of its camera frame at a specific size.
    
    
    """
    def __init__(self, cfg: Optional[EnvConfig] = None, trajectory_mode: str = "triangular"):
        super().__init__() # Call the parent class constructor to ensure proper initialization
        self.cfg = cfg if cfg is not None else EnvConfig() # Use provided config or default values
        self.rng = np.random.default_rng(self.cfg.seed) # Initialize random number generator for reproducibility

        # Action: [v_x, v_y] relative velocity commands
        self.action_space = gym.spaces.Box(low=-self.cfg.max_action, high=self.cfg.max_action, shape=(2,), dtype=np.float32)
        # Obs: [x_error, size_error] - visual state relative to target
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)


        # Setup PyBullet physics engine in DIRECT (headless) mode
        self._client = p.connect(p.DIRECT) # Use p.GUI for visualization during development, but DIRECT is faster for training
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) # To load plane.urdf and other assets
        p.setGravity(0.0, 0.0, -9.81) 

        # Load environment assets
        self.plane_id = p.loadURDF("plane.urdf")
        self.drone_id = self._create_drone_body()
        self.target_id = self._create_target_body()

        self.occlusion_wall_ids = []
        self.occlusion_walls_xy = [] # Stores (p1, p2, height) for fast LoS checks
        self._create_occlusions_if_needed(trajectory_mode)

        self.trajectory = TargetTrajectory(trajectory_mode)
        self.vision = VisionBBoxEstimator(self.cfg.vision, self.rng)

    def _create_drone_body(self) -> int:
        """Constructs a blue box representing the drone."""
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.08])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.25, 0.25, 0.08], rgbaColor=[0.2, 0.5, 0.9, 1.0])
        return p.createMultiBody(baseMass=1.2, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=[30, 60, self.cfg.drone_altitude])

    def _create_target_body(self) -> int:
        """Constructs a red box representing the ground target."""
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.8])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.8], rgbaColor=[0.9, 0.2, 0.2, 1.0])
        return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)

    def _create_occlusions_if_needed(self, mode: str):
        """Adds walls for the 'square_wave' path to test tracking through occlusions."""
        if mode != "square_wave": return
        wall_centers = [np.array([55, 95, 2.5]), np.array([105, 145, 2.5]), np.array([155, 105, 2.5])]
        for c in wall_centers:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 4.0, 2.5])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 4.0, 2.5], rgbaColor=[0.4, 0.4, 0.4, 0.8])
            self.occlusion_wall_ids.append(p.createMultiBody(0, col, vis, c.tolist()))
            self.occlusion_walls_xy.append((np.array([c[0], c[1]-4]), np.array([c[0], c[1]+4]), 5.0))

    def reset(self, *, seed=None, options=None):
        """Resets the environment to the starting state for a new episode."""
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.step_count, self.t = 0, 0.0
        self.vision.reset()

        # Initialize positions
        self.drone_pos = np.array([30.0, 60.0, self.cfg.drone_altitude], dtype=np.float32)
        self.drone_vel = np.zeros(3, dtype=np.float32)
        self.target_pos, _ = self.trajectory.sample(0.0)

        # Update PyBullet
        p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

        # Get initial observation
        bbox = self._observe_bbox()
        state = vision_state_from_bbox(bbox["x_center"], bbox["area"], self.cfg.x_des, self.cfg.s_des)
        self.prev_state = state.copy()

        return state, {"bbox": bbox}

    def step(self, action: np.ndarray):
        """Processes agent action, updates physics, and returns (obs, reward, done, info)."""
        action = np.clip(action, -self.cfg.max_action, self.cfg.max_action)

        # 1. Update Physics
        self._apply_drone_control(action)
        self._advance_target(self.cfg.control_period)
        
        # Simulating at a higher frequency than control frequency for numerical stability
        n_substeps = max(1, int(round(self.cfg.control_period / self.cfg.sim_dt)))
        for _ in range(n_substeps): p.stepSimulation()

        # 2. Perception
        bbox = self._observe_bbox()
        state = vision_state_from_bbox(bbox["x_center"], bbox["area"], self.cfg.x_des, self.cfg.s_des)

        # 3. Reward and Termination
        reward, reward_terms = self._compute_reward(state, action)
        self.prev_state, self.step_count, self.t = state.copy(), self.step_count + 1, self.t + self.cfg.control_period

        out_of_bounds = not (0 < self.drone_pos[0] < self.cfg.world_x and 0 < self.drone_pos[1] < self.cfg.world_y)
        truncated = self.step_count >= self.cfg.max_episode_steps or out_of_bounds

        return state, float(reward), False, truncated, {"reward_terms": reward_terms, "out_of_bounds": out_of_bounds}

    def _apply_drone_control(self, action_xy: np.ndarray):
        """
        Translates RL velocity commands into physics-based movement.
        Applies acceleration limits and drag.
        """
        desired_vel = np.array([action_xy[0], action_xy[1], 0.0], dtype=np.float32)
        accel_cmd = desired_vel - self.drone_vel

        # Enforce physical acceleration limits
        accel_norm = np.linalg.norm(accel_cmd[:2])
        if accel_norm > self.cfg.max_accel:
            accel_cmd[:2] *= (self.cfg.max_accel / (accel_norm + 1e-8))

        # Update velocity and position (Euler integration)
        self.drone_vel += accel_cmd * self.cfg.control_period
        self.drone_vel[:2] *= (1.0 - self.cfg.drag * self.cfg.control_period)
        self.drone_pos += self.drone_vel * self.cfg.control_period
        self.drone_pos[2] = self.cfg.drone_altitude # Maintain fixed height

        p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
        p.resetBaseVelocity(self.drone_id, self.drone_vel.tolist())

    def _advance_target(self, dt: float):
        """Moves target in PyBullet based on the trajectory class."""
        self.target_pos, self.target_vel = self.trajectory.sample(self.t + dt)
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

    def _observe_bbox(self) -> Dict[str, float]:
        """Calculates simulated camera bounding box, considering occlusions."""
        occluded = self._is_occluded(self.drone_pos, self.target_pos)
        return self.vision.project_bbox(self.drone_pos, 0.0, self.target_pos, occluded=occluded)

    def _is_occluded(self, a: np.ndarray, b: np.ndarray) -> bool:
        """
        Fast Line-of-Sight (LoS) check. Returns True if a wall is between 
        the drone (a) and the target (b).
        """
        if not self.occlusion_walls_xy: return False
        p1, p2 = a[:2], b[:2]
        for w1, w2, wall_h in self.occlusion_walls_xy:
            if max(a[2], b[2]) > wall_h: continue # Drone flies over the wall
            if _segments_intersect(p1, p2, w1, w2): return True
        return False

    def _compute_reward(self, s_t: np.ndarray, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Calculates the Reward based on:
        1. Rs: Error reduction (tracking accuracy).
        2. Rspeed: Incentive to move fast when far, but slow when close.
        3. Rstability: Reward for keeping the target centered with minimal jitter.
        """
        rs = (abs(self.prev_state[0]) - abs(s_t[0])) + (abs(self.prev_state[1]) - abs(s_t[1]))
        v_mag = float(np.linalg.norm(action))
        
        # If far away, reward higher speed to close the gap
        far = (abs(s_t[0]) > 1.5) or (abs(s_t[1]) > 1.5)
        rspeed = v_mag if far else -v_mag

        # If perfectly centered, reward low velocity (stable hover)
        stable = (abs(s_t[0]) < 0.015) and (abs(action[0]) < 0.1)
        rstability = v_mag if stable else -v_mag

        reward = self.cfg.w1 * rs + self.cfg.w2 * rspeed + self.cfg.w3 * rstability
        return reward, {"Rs": rs, "Rspeed": rspeed, "Rstability": rstability}

    def close(self):
        """Disconnects the physics client."""
        if p.isConnected(self._client): p.disconnect(self._client)


# --- GEOMETRY UTILITIES FOR OCCLUSION CHECK ---

def _orientation(a, b, c):
    """Returns the orientation of an ordered triplet (a,b,c). 0=collinear, 1=CW, 2=CCW."""
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    return 0 if abs(val) < 1e-8 else (1 if val > 0 else 2)

def _on_segment(a, b, c):
    """Checks if point b lies on line segment 'ac'."""
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])

def _segments_intersect(p1, q1, p2, q2):
    """Returns True if line segment 'p1q1' and 'p2q2' intersect."""
    o1, o2, o3, o4 = _orientation(p1, q1, p2), _orientation(p1, q1, q2), _orientation(p2, q2, p1), _orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and _on_segment(p1, p2, q1): return True
    if o2 == 0 and _on_segment(p1, q2, q1): return True
    if o3 == 0 and _on_segment(p2, p1, q2): return True
    if o4 == 0 and _on_segment(p2, q1, q2): return True
    return False