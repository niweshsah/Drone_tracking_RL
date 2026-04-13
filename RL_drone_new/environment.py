import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Tuple

# Below has:
# init( start_xy: Tuple[float, float], scale: float = 50.0, mode: str = "triangular") 
# sample(t: float) -> Tuple[np.ndarray, np.ndarray] which returns position and velocity at time t
from target_trajectory import TargetTrajectory 



class DroneTrackingEnv(gym.Env):
    def __init__(self, cfg=None, trajectory_mode: str = "triangular", GUI_mode: bool = False):
        super().__init__()
        
        # Enhanced config for Feedforward/Residual Control
        self.cfg = cfg if cfg is not None else type('obj', (object,), {
            'seed': 42, 
            'max_action': 5.0,          # Absolute max velocity of drone (m/s)
            'max_residual_vel': 2.0,    # Max velocity CORRECTION the RL agent can apply
            'target_vel_noise': 0.2,    # Standard deviation of BoT-SORT velocity estimation noise
            'max_accel': 10.0,          # Physics acceleration limit
            'drag': 0.1,                # Air resistance
            'drone_altitude': 5.0, 
            'control_period': 0.05, 
            'sim_dt': 1/240,
            'world_x': 500, 'world_y': 500, 'max_episode_steps': 1000,
            'drone_start': [0, 0], 'target_start': [0, 0], 'trajectory_scale': 50.0,
            's_des': 0.05, 'vision': None,
            'reward_weights': {
                'align': 2.0, 'scale': -1.5, 'smooth': -0.2, 'energy': -0.05, 'boundary': -1.0, 'crash': -50.0
            }
            
        })
        
        self.GUI_mode = GUI_mode
        
        
        self.rng = np.random.default_rng(self.cfg.seed)

        # Action: [delta_vx, delta_vy] normalized residual velocity commands [-1, 1]
        # gym.spaces.Box is used to define continuous action and observation spaces.
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Obs: [x_norm, y_norm, w_norm, h_norm, drone_vx_norm, drone_vy_norm]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._client = p.connect(p.DIRECT) if not GUI_mode else p.connect(p.GUI)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        p.setGravity(0.0, 0.0, -9.81) 

        self.plane_id = p.loadURDF("plane.urdf") # Ground plane for visual reference
        self.drone_id = self._create_drone_body() # spherical drone body
        self.target_id = self._create_target_body() # box target body

        # self.trajectory = TargetTrajectory(trajectory_mode, self.cfg.target_start, self.cfg.trajectory_scale)
        self.trajectory = TargetTrajectory(
            start_xy=self.cfg.target_start, 
            scale=self.cfg.trajectory_scale, 
            mode=trajectory_mode
        )
        self.prev_action = np.zeros(2, dtype=np.float32)

    def _create_drone_body(self) -> int:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.3, rgbaColor=[0.2, 0.5, 0.9, 1.0])
        return p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, 
                                 basePosition=[self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude])

    def _create_target_body(self) -> int:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.2])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.2], rgbaColor=[0.9, 0.2, 0.2, 1.0])
        return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)
    
    def _update_gui_camera(self):
        """Updates the PyBullet GUI camera to follow the drone."""
        
        # Calculate the drone's movement direction (Yaw)
        # If the drone is moving, point the camera in the direction of velocity
        speed = np.linalg.norm(self.drone_vel[:2])
        if speed > 0.1:
            # math.atan2(y, x) gives the angle in radians, convert to degrees
            yaw = np.degrees(np.arctan2(self.drone_vel[1], self.drone_vel[0]))
        else:
            # If hovering/stopped, just default to 0 or keep the previous yaw
            yaw = getattr(self, "last_camera_yaw", 0)
            
        self.last_camera_yaw = yaw

        # Set camera parameters
        camera_distance = 6.0      # How far back from the drone
        camera_pitch = -25.0       # Angle looking down at the drone
        
        # The camera focuses on the drone's exact position
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=yaw - 90, # PyBullet yaw offset adjustment
            cameraPitch=camera_pitch,
            cameraTargetPosition=self.drone_pos.tolist(),
            physicsClientId=self._client
        )

    def _get_obs(self, bbox: Dict[str, float]) -> np.ndarray:
        # Normalize ego-velocities by absolute max_action
        vx_norm = np.clip(self.drone_vel[0] / self.cfg.max_action, -1, 1)
        vy_norm = np.clip(self.drone_vel[1] / self.cfg.max_action, -1, 1)

        return np.array([
            bbox.get("x_center", 0.0), 
            bbox.get("y_center", 0.0), 
            bbox.get("width", 0.0), 
            bbox.get("height", 0.0), 
            vx_norm, 
            vy_norm
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.step_count, self.t = 0, 0.0 
        self.prev_action = np.zeros(2, dtype=np.float32) # Reset previous action for smoothness penalty

        self.drone_pos = np.array([self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude], dtype=np.float32) # bring back the drone to its initial position
        self.drone_vel = np.zeros(3, dtype=np.float32) # set initial velocity to zero for consistency
        
        # Unpack both position AND velocity
        self.target_pos, self.target_vel = self.trajectory.sample(0.0)

        p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

        bbox = self._observe_bbox()
        return self._get_obs(bbox), {"bbox": bbox}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        
        # --- THE NEW RESIDUAL CONTROL PARADIGM ---
        # 1. Simulate BoT-SORT velocity estimation (True Velocity + Gaussian Noise)
        # right now, the target velocity is directly returned by the trajectory, but in a real scenario, this would come from a vision-based tracker like BoT-SORT. We simulate this by adding noise to the true velocity.
        estimated_target_vel = self.target_vel[:2] + self.rng.normal(0, self.cfg.target_vel_noise, size=2)
        
        residual_vel = action * self.cfg.max_residual_vel
        
        # 3. Final commanded velocity = Feedforward + Residual
        desired_vel_xy = estimated_target_vel + residual_vel
        
        # 4. Enforce absolute physical limits
        desired_vel_xy = np.clip(desired_vel_xy, -self.cfg.max_action, self.cfg.max_action)
        # -----------------------------------------

        # 5. Physics Update
        self._apply_drone_control(desired_vel_xy)
        self._advance_target(self.cfg.control_period)
        
        n_substeps = max(1, int(round(self.cfg.control_period / self.cfg.sim_dt)))
        for _ in range(n_substeps): p.stepSimulation()
        
        if self.GUI_mode:
            self._update_gui_camera()

        # 6. Perception & Terminations
        bbox = self._observe_bbox()
        state = self._get_obs(bbox)

        target_lost = (abs(state[0]) > 0.98 or abs(state[1]) > 0.98 or bbox["area"] <= 0)
        out_of_bounds = not (-self.cfg.world_x < self.drone_pos[0] < self.cfg.world_x 
                             and -self.cfg.world_y < self.drone_pos[1] < self.cfg.world_y)
        
        # 7. Reward Calculation
        reward, reward_terms = self._compute_reward(state, action, target_lost)

        self.step_count += 1
        self.t += self.cfg.control_period
        self.prev_action = action.copy()

        # terminated = target_lost
        terminated = False # For this version, we won't consider target loss as a terminal condition, but this can be adjusted based on desired behavior
        truncated = self.step_count >= self.cfg.max_episode_steps or out_of_bounds

        return state, float(reward), terminated, truncated, {"reward_terms": reward_terms, "bbox": bbox}

    def _apply_drone_control(self, desired_vel_xy: np.ndarray):
        accel_cmd = (desired_vel_xy - self.drone_vel[:2]) / self.cfg.control_period
        
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.cfg.max_accel:
            accel_cmd *= (self.cfg.max_accel / accel_mag)

        self.drone_vel[:2] += accel_cmd * self.cfg.control_period
        self.drone_vel[:2] *= (1.0 - self.cfg.drag * self.cfg.control_period)
        
        self.drone_pos[:2] += self.drone_vel[:2] * self.cfg.control_period
        self.drone_pos[2] = self.cfg.drone_altitude 

        p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
        p.resetBaseVelocity(self.drone_id, self.drone_vel.tolist())


    # Reward function now includes:
    # 1. Alignment reward based on distance to target center: r_align = 2.0 * exp(-5.0 * dist_err)
    # 2. Scale reward based on how well the observed area matches desired size: r_scale = -1.5 * abs((w_n * h_n) - s_des)
    # 3. Smoothness penalty on changes in the RESIDUAL action (encouraging smoother corrections): r_smooth = -0.2 * sum((action - prev_action)^2)
    # 4. Optional energy penalty to encourage minimal corrections when well-aligned
    # 5. Boundary penalty for being too far from the target
    # 6. Crash penalty for losing the target
    
    def _compute_reward(self, state: np.ndarray, action: np.ndarray, target_lost: bool) -> Tuple[float, Dict]:
        
        x_n, y_n, w_n, h_n, _, _ = state # Unpack normalized bbox and velocities
        
        dist_err = np.sqrt(x_n**2 + y_n**2) # Distance error in normalized image space
        
        wt_dict = self.cfg.reward_weights
        w1 , w2, w3, w4, w5, w6 = wt_dict['align'], wt_dict['scale'], wt_dict['smooth'], wt_dict['energy'], wt_dict['boundary'], wt_dict['crash']
        
        r_align = w1 * np.exp(-5.0 * dist_err) # Strong exponential reward for being close to the target center
        
        r_scale = 0 * w2
        # r_scale = -1.5 * abs((w_n * h_n) - self.cfg.s_des) # Penalize deviation from desired scale (size in image)
        # not needed as drone has fix altitude
        
        # The Smoothness penalty now punishes high-frequency changes in the RESIDUAL action
        r_smooth = w3 * np.sum(np.square(action - self.prev_action)) # Encourage smoother corrections by penalizing large changes in the residual action from one step to the next
        
        # Optional: Add a small energy penalty to incentivize the agent to output [0,0] when perfectly aligned
        r_energy = w4 * np.sum(np.square(action)) # Penalize large corrections to encourage minimal intervention when well-aligned

        r_boundary = 0 * w5 * max(0, dist_err - 0.75) # not considering boundary penalty in this version, but can be adjusted based on desired behavior
        r_crash = w6 * 0 if target_lost else 0.0 # not  considering target loss as a crash in this version, but can be adjusted based on desired behavior

        total = r_align + r_scale + r_smooth + r_energy + r_boundary + r_crash
        return total, {"align": r_align, "smooth": r_smooth, "crash": r_crash}

    def _observe_bbox(self) -> Dict[str, float]:
        rel_pos = self.target_pos - self.drone_pos
        x_img = np.clip(rel_pos[0] / 10.0, -1, 1) 
        y_img = np.clip(rel_pos[1] / 10.0, -1, 1)
        area = np.clip(1.0 / (np.linalg.norm(rel_pos) + 1e-5), 0, 1)
        return {"x_center": x_img, "y_center": y_img, "width": 0.1, "height": 0.1, "area": area}

    def _advance_target(self, dt: float):
        # Now explicitly storing the velocity returned by the trajectory
        self.target_pos, self.target_vel = self.trajectory.sample(self.t + dt)
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

    def close(self):
        if p.isConnected(self._client): p.disconnect(self._client)