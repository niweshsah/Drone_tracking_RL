import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Tuple
import math

# Below has:
# init( start_xy: Tuple[float, float], scale: float = 50.0, mode: str = "triangular") 
# sample(t: float) -> Tuple[np.ndarray, np.ndarray] which returns position and velocity at time t
from target_trajectory import TargetTrajectory

Drone_obj_path = "./Drone_Costum/Material/drone_costum.obj"



from rewards import RewardDrone

class DroneTrackingEnv(gym.Env):
    def __init__(self, cfg=None, trajectory_mode: str = "triangular", GUI_mode: bool = False):
        super().__init__()
        
        # Enhanced config for Feedforward/Residual Control
        self.cfg = cfg if cfg is not None else type('obj', (object,), {
            'seed': 42, 
            'max_action': 15.0,          # Absolute max velocity of drone (m/s)  for each axis
            'max_residual_vel': 2.0,    # Max velocity CORRECTION the RL agent can apply
            'target_vel_noise': 0.2,    # Standard deviation of BoT-SORT velocity estimation noise
            'max_accel': 10.0,          # Physics acceleration limit
            'drag': 0.1,                # Air resistance
            'drone_altitude': 5.0, 
            'control_period': 0.05, 
            'sim_dt': 1/240,
            'world_x': 5000, 'world_y': 5000, 'max_episode_steps': 1000,
            'drone_start': [0, 0], 'target_start': [0, 0], 'trajectory_scale': 50.0,
            's_des': 0.05, 'vision': None,
            'drone_size' : 0.1 # b/w 0 and 1
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
        
        # Compute reward(state, action) -> total reward, reward_terms
        self._compute_reward = RewardDrone()._compute_reward

    def _create_drone_body(self) -> int:
        """
        Maintains the 0.3 radius sphere collision for identical dynamics, 
        but maps a high-fidelity 3D mesh for visuals.
        """
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        visual_orientation = p.getQuaternionFromEuler([math.pi/2, 0 , 0])
        drone_size = self.cfg.drone_size

        try:
            vis = p.createVisualShape(p.GEOM_MESH, fileName= Drone_obj_path, meshScale=[drone_size, drone_size, drone_size],  visualFrameOrientation=visual_orientation)
        except p.error:
            # Fallback placeholder if no OBJ is found
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.1], rgbaColor=[0.1, 0.1, 0.1, 1.0])

        return p.createMultiBody(
            baseMass=1.0, 
            baseCollisionShapeIndex=col, 
            baseVisualShapeIndex=vis, 
            basePosition=[self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude]
        )

    # def _create_target_body(self) -> int:
    #     col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.2])
    #     vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.2], rgbaColor=[0.9, 0.2, 0.2, 1.0])
    #     return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)
    
    def _create_target_body(self) -> int:
        """Loads a Husky rover model from pybullet_data as the target instead of a box."""
        # Using a built-in rover model. We set useFixedBase=True because we are 
        # kinematically teleporting it via self.trajectory.sample(), not driving its wheels.
        target_id = p.loadURDF("husky/husky.urdf", useFixedBase=True)
        return target_id
    
    def _update_gui_camera(self):
        """Updates the PyBullet GUI camera to smoothly follow the drone."""
        
        try:
            # Grab the current camera settings (so we don't override your mouse movements!)
            cam_info = p.getDebugVisualizerCamera(physicsClientId=self._client)
            current_yaw = cam_info[8]
            current_pitch = cam_info[9]
            current_dist = cam_info[10]
        except Exception:
            # Fallback defaults if the camera hasn't initialized yet
            current_yaw = 45.0
            current_pitch = -35.0
            current_dist = 6.0

        # Update ONLY the target position to follow the drone. 
        # The camera will glide smoothly without rotating violently.
        p.resetDebugVisualizerCamera(
            cameraDistance=current_dist,
            cameraYaw=current_yaw,
            cameraPitch=current_pitch,
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

        target_lost = (abs(state[0]) > 0.98 or abs(state[1]) > 0.98 or bbox["area"] <= 0) # if xn or yn is near the edge of the image or if the target is not visible at all, we consider it lost
        out_of_bounds = not (-self.cfg.world_x < self.drone_pos[0] < self.cfg.world_x 
                             and -self.cfg.world_y < self.drone_pos[1] < self.cfg.world_y)
        
        # 7. Reward Calculation
        reward, reward_terms = self._compute_reward(state, action, target_lost, self.prev_action)

        self.step_count += 1
        self.t += self.cfg.control_period
        self.prev_action = action.copy()

        terminated = target_lost
        # terminated = False # For this version, we won't consider target loss as a terminal condition, but this can be adjusted based on desired behavior
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

    def _observe_bbox(self) -> Dict[str, float]:
        rel_pos = self.target_pos - self.drone_pos
        x_img = np.clip(rel_pos[0] / 10.0, -1, 1) 
        y_img = np.clip(rel_pos[1] / 10.0, -1, 1)
        area = np.clip(1.0 / (np.linalg.norm(rel_pos) + 1e-5), 0, 1)
        return {"x_center": x_img, "y_center": y_img, "width": 0.1, "height": 0.1, "area": area}

    def _advance_target(self, dt: float):
        # Now explicitly storing the velocity returned by the trajectory
        self.target_pos, self.target_vel = self.trajectory.sample(self.t + dt)
        
        # --- DYNAMIC HUSKY ROTATION ---
        target_speed = np.linalg.norm(self.target_vel[:2])
        
        # If moving, calculate the angle of travel. If stopped, keep facing the same way.
        if target_speed > 0.01:
            yaw = math.atan2(self.target_vel[1], self.target_vel[0])
        else:
            yaw = getattr(self, "husky_yaw", 0.0)
        self.husky_yaw = yaw

        # The Husky stays flat on the ground, so Roll=0 and Pitch=0
        target_orientation = p.getQuaternionFromEuler([0, 0, yaw])
        # ------------------------------

        # Apply the position and the calculated orientation
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), target_orientation)

    def close(self):
        if p.isConnected(self._client): p.disconnect(self._client)