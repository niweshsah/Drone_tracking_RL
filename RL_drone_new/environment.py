# # import gymnasium as gym
# # import numpy as np
# # import pybullet as p
# # import pybullet_data
# # from typing import Dict, Tuple
# # import math
# # import os

# # # Below has:
# # # init( start_xy: Tuple[float, float], scale: float = 50.0, mode: str = "triangular") 
# # # sample(t: float) -> Tuple[np.ndarray, np.ndarray] which returns position and velocity at time t
# # from target_trajectory import TargetTrajectory
# # from rewards import RewardDrone

# # ENV_DIR = os.path.dirname(os.path.abspath(__file__))

# # # Safely construct the paths to your assets
# # Drone_obj_path = os.path.join(ENV_DIR, "Drone_Costum", "Material", "drone_costum.obj")
# # TEXTURE_PATH = os.path.join(ENV_DIR, "grass", "textures", "coast_sand_rocks_02_diff_4k.jpg")


# # class DroneTrackingEnv(gym.Env):
# #     def __init__(self, cfg=None, trajectory_mode: str = "triangular", GUI_mode: bool = False):
# #         super().__init__()
        
# #         # Enhanced config for Feedforward/Residual Control
# #         self.cfg = cfg if cfg is not None else type('obj', (object,), {
# #             'seed': 42, 
# #             'max_action': 15.0,         # Absolute max velocity of drone (m/s)  for each axis
# #             'max_residual_vel': 2.0,    # Max velocity CORRECTION the RL agent can apply
# #             'target_vel_noise': 0.2,    # Standard deviation of BoT-SORT velocity estimation noise
# #             'max_accel': 10.0,          # Physics acceleration limit
# #             'drag': 0.1,                # Air resistance
# #             'drone_altitude': 5.0, 
# #             'control_period': 0.05, 
# #             'sim_dt': 1/240,
# #             'world_x': 5000, 'world_y': 5000, 'max_episode_steps': 1000,
# #             'drone_start': [0, 0], 'target_start': [0, 0], 'trajectory_scale': 40.0,
# #             's_des': 0.05, 'vision': None,
# #             'drone_size' : 0.1 # b/w 0 and 1
# #         })
        
# #         self.GUI_mode = GUI_mode
# #         self.default_trajectory_mode = trajectory_mode
        
        
# #         self.rng = np.random.default_rng(self.cfg.seed)

# #         # Action: [delta_vx, delta_vy] normalized residual velocity commands [-1, 1]
# #         # gym.spaces.Box is used to define continuous action and observation spaces.
# #         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
# #         # Obs: [x_norm, y_norm, w_norm, h_norm, drone_vx_norm, drone_vy_norm]
# #         self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

# #         self._client = p.connect(p.GUI) if self.GUI_mode else p.connect(p.DIRECT)
        
# #         p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
# #         p.setGravity(0.0, 0.0, -9.81) 

# #         self.plane_id = p.loadURDF("plane.urdf") # Ground plane for visual reference
        
# #         # Load your realistic image texture
# #         texture_id = p.loadTexture(TEXTURE_PATH)
        
# #         # Apply the texture to the base of the plane (linkIndex -1)
# #         p.changeVisualShape(
# #             self.plane_id, 
# #             -1, 
# #             textureUniqueId=texture_id,
# #             rgbaColor=[1, 1, 1, 1], # Keep base color white so it doesn't tint the image
# #             specularColor=[0.1, 0.1, 0.1] # Turn down shininess (default is very glossy)
# #         )
        
# #         if GUI_mode:
# #             p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
# #             p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
# #             # Set a nice "afternoon" sun angle for good shadows
# #             p.configureDebugVisualizer(lightPosition=[10, 10, 10])
            
            
# #         self.drone_id = self._create_drone_body() # spherical drone body
# #         self.target_id = self._create_target_body() # box target body

# #         # self.trajectory = TargetTrajectory(trajectory_mode, self.cfg.target_start, self.cfg.trajectory_scale)
# #         self.trajectory = TargetTrajectory(
# #             start_xy=self.cfg.target_start, 
# #             scale=self.cfg.trajectory_scale, 
# #             mode=trajectory_mode
# #         )
# #         self.prev_action = np.zeros(2, dtype=np.float32)
        
# #         # Compute reward(state, action) -> total reward, reward_terms
# #         self._compute_reward = RewardDrone()._compute_reward

# #     def _create_drone_body(self) -> int:
# #         """
# #         Maintains the 0.3 radius sphere collision for identical dynamics, 
# #         but maps a high-fidelity 3D mesh for visuals.
# #         """
# #         col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
# #         visual_orientation = p.getQuaternionFromEuler([math.pi/2, 0 , 0])
# #         drone_size = self.cfg.drone_size

# #         try:
# #             vis = p.createVisualShape(p.GEOM_MESH, fileName= Drone_obj_path, meshScale=[drone_size, drone_size, drone_size],  visualFrameOrientation=visual_orientation)
# #         except p.error:
# #             # Fallback placeholder if no OBJ is found
# #             vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.1], rgbaColor=[0.1, 0.1, 0.1, 1.0])

# #         return p.createMultiBody(
# #             baseMass=1.0, 
# #             baseCollisionShapeIndex=col, 
# #             baseVisualShapeIndex=vis, 
# #             basePosition=[self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude]
# #         )

# #     # def _create_target_body(self) -> int:
# #     #     col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.2])
# #     #     vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.2], rgbaColor=[0.9, 0.2, 0.2, 1.0])
# #     #     return p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)
    
# #     def _create_target_body(self) -> int:
# #         """Loads a Husky rover model from pybullet_data as the target instead of a box."""
# #         # Using a built-in rover model. We set useFixedBase=True because we are 
# #         # kinematically teleporting it via self.trajectory.sample(), not driving its wheels.
# #         target_id = p.loadURDF("husky/husky.urdf", useFixedBase=True)
# #         return target_id
    
# #     def _update_gui_camera(self):
# #         """Updates the PyBullet GUI camera to smoothly follow the drone."""
        
# #         try:
# #             # Grab the current camera settings (so we don't override your mouse movements!)
# #             cam_info = p.getDebugVisualizerCamera(physicsClientId=self._client)
# #             current_yaw = cam_info[8]
# #             current_pitch = cam_info[9]
# #             current_dist = cam_info[10]
# #         except Exception:
# #             # Fallback defaults if the camera hasn't initialized yet
# #             current_yaw = 45.0
# #             current_pitch = -35.0
# #             current_dist = 6.0

# #         # Update ONLY the target position to follow the drone. 
# #         # The camera will glide smoothly without rotating violently.
# #         p.resetDebugVisualizerCamera(
# #             cameraDistance=current_dist,
# #             cameraYaw=current_yaw,
# #             cameraPitch=current_pitch,
# #             cameraTargetPosition=self.drone_pos.tolist(),
# #             physicsClientId=self._client
# #         )

# #     def _get_obs(self, bbox: Dict[str, float]) -> np.ndarray:
# #         # Normalize ego-velocities by absolute max_action
# #         vx_norm = np.clip(self.drone_vel[0] / self.cfg.max_action, -1, 1)
# #         vy_norm = np.clip(self.drone_vel[1] / self.cfg.max_action, -1, 1)

# #         return np.array([
# #             bbox.get("x_center", 0.0), 
# #             bbox.get("y_center", 0.0), 
# #             bbox.get("width", 0.0), 
# #             bbox.get("height", 0.0), 
# #             vx_norm, 
# #             vy_norm
# #         ], dtype=np.float32)

# #     # def reset(self, *, seed=None, options=None):
# #     #     if seed is not None: self.rng = np.random.default_rng(seed)
# #     #     self.step_count, self.t = 0, 0.0 
# #     #     self.prev_action = np.zeros(2, dtype=np.float32) # Reset previous action for smoothness penalty

# #     #     self.drone_pos = np.array([self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude], dtype=np.float32) # bring back the drone to its initial position
# #     #     self.drone_vel = np.zeros(3, dtype=np.float32) # set initial velocity to zero for consistency
        
# #     #     # Unpack both position AND velocity
# #     #     self.target_pos, self.target_vel = self.trajectory.sample(0.0)

# #     #     p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
# #     #     p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

# #     #     bbox = self._observe_bbox()
# #     #     return self._get_obs(bbox), {"bbox": bbox}
    
# #     def reset(self, *, seed=None, options=None):
# #         # 1. Standard Gym Seeding
# #         super().reset(seed=seed)
# #         if seed is not None: 
# #             self.rng = np.random.default_rng(seed)
            
# #         self.step_count, self.t = 0, 0.0 
# #         self.prev_action = np.zeros(2, dtype=np.float32)

# #         # 2. Extract trajectory_mode from options (standard Gymnasium pattern)
# #         traj_mode = self.default_trajectory_mode
# #         if options is not None and "trajectory_mode" in options:
# #             traj_mode = options["trajectory_mode"]

# #         # 3. Instantiate the hot-swapped trajectory
# #         self.trajectory = TargetTrajectory(
# #             start_xy=self.cfg.target_start, 
# #             scale=self.cfg.trajectory_scale, 
# #             mode=traj_mode
# #         )

# #         # 4. Reset physical drone state
# #         self.drone_pos = np.array([self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude], dtype=np.float32)
# #         self.drone_vel = np.zeros(3, dtype=np.float32)
        
# #         # Unpack both position AND velocity from the newly selected trajectory
# #         self.target_pos, self.target_vel = self.trajectory.sample(0.0)

# #         p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
# #         p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

# #         # 5. Return Observation and Info dict
# #         bbox = self._observe_bbox()
        
# #         # Include current trajectory in info to ensure train.py logging works perfectly
# #         info = {
# #             "bbox": bbox, 
# #             "current_trajectory": traj_mode
# #         }
        
# #         return self._get_obs(bbox), info

# #     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
# #         action = np.clip(action, -1.0, 1.0)
        
# #         # --- THE NEW RESIDUAL CONTROL PARADIGM ---
# #         # 1. Simulate BoT-SORT velocity estimation (True Velocity + Gaussian Noise)
# #         # right now, the target velocity is directly returned by the trajectory, but in a real scenario, this would come from a vision-based tracker like BoT-SORT. We simulate this by adding noise to the true velocity.
# #         estimated_target_vel = self.target_vel[:2] + self.rng.normal(0, self.cfg.target_vel_noise, size=2)
        
# #         residual_vel = action * self.cfg.max_residual_vel
        
# #         # 3. Final commanded velocity = Feedforward + Residual
# #         desired_vel_xy = estimated_target_vel + residual_vel
        
# #         # 4. Enforce absolute physical limits
# #         desired_vel_xy = np.clip(desired_vel_xy, -self.cfg.max_action, self.cfg.max_action)
# #         # -----------------------------------------

# #         # 5. Physics Update
# #         self._apply_drone_control(desired_vel_xy)
# #         self._advance_target(self.cfg.control_period)
        
# #         n_substeps = max(1, int(round(self.cfg.control_period / self.cfg.sim_dt)))
# #         for _ in range(n_substeps): p.stepSimulation()
        
# #         if self.GUI_mode:
# #             self._update_gui_camera()

# #         # 6. Perception & Terminations
# #         bbox = self._observe_bbox()
# #         state = self._get_obs(bbox)

# #         target_lost = (abs(state[0]) > 0.98 or abs(state[1]) > 0.98 or bbox["area"] <= 0) # if xn or yn is near the edge of the image or if the target is not visible at all, we consider it lost
# #         out_of_bounds = not (-self.cfg.world_x < self.drone_pos[0] < self.cfg.world_x 
# #                              and -self.cfg.world_y < self.drone_pos[1] < self.cfg.world_y)
        
# #         # 7. Reward Calculation
# #         reward, reward_terms = self._compute_reward(state, action, target_lost, self.prev_action)

# #         self.step_count += 1
# #         self.t += self.cfg.control_period
# #         self.prev_action = action.copy()

# #         terminated = target_lost
# #         # terminated = False # For this version, we won't consider target loss as a terminal condition, but this can be adjusted based on desired behavior
# #         truncated = self.step_count >= self.cfg.max_episode_steps or out_of_bounds

# #         return state, float(reward), terminated, truncated, {"reward_terms": reward_terms, "bbox": bbox}

# #     def _apply_drone_control(self, desired_vel_xy: np.ndarray):
# #         accel_cmd = (desired_vel_xy - self.drone_vel[:2]) / self.cfg.control_period
        
# #         accel_mag = np.linalg.norm(accel_cmd)
# #         if accel_mag > self.cfg.max_accel:
# #             accel_cmd *= (self.cfg.max_accel / accel_mag)

# #         self.drone_vel[:2] += accel_cmd * self.cfg.control_period
# #         self.drone_vel[:2] *= (1.0 - self.cfg.drag * self.cfg.control_period)
        
# #         self.drone_pos[:2] += self.drone_vel[:2] * self.cfg.control_period
# #         self.drone_pos[2] = self.cfg.drone_altitude 

# #         p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
# #         p.resetBaseVelocity(self.drone_id, self.drone_vel.tolist())

# #     def _observe_bbox(self) -> Dict[str, float]:
# #         rel_pos = self.target_pos - self.drone_pos
# #         x_img = np.clip(rel_pos[0] / 10.0, -1, 1) 
# #         y_img = np.clip(rel_pos[1] / 10.0, -1, 1)
# #         area = np.clip(1.0 / (np.linalg.norm(rel_pos) + 1e-5), 0, 1)
# #         return {"x_center": x_img, "y_center": y_img, "width": 0.1, "height": 0.1, "area": area}

# #     def _advance_target(self, dt: float):
# #         # Now explicitly storing the velocity returned by the trajectory
# #         self.target_pos, self.target_vel = self.trajectory.sample(self.t + dt)
        
# #         # --- DYNAMIC HUSKY ROTATION ---
# #         target_speed = np.linalg.norm(self.target_vel[:2])
        
# #         # If moving, calculate the angle of travel. If stopped, keep facing the same way.
# #         if target_speed > 0.01:
# #             yaw = math.atan2(self.target_vel[1], self.target_vel[0])
# #         else:
# #             yaw = getattr(self, "husky_yaw", 0.0)
# #         self.husky_yaw = yaw

# #         # The Husky stays flat on the ground, so Roll=0 and Pitch=0
# #         target_orientation = p.getQuaternionFromEuler([0, 0, yaw])
# #         # ------------------------------

# #         # Apply the position and the calculated orientation
# #         p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), target_orientation)

# #     def close(self):
# #         if p.isConnected(self._client): p.disconnect(self._client)

























# import gymnasium as gym
# import numpy as np
# import pybullet as p
# import pybullet_data
# from typing import Dict, Tuple
# import math
# import os

# # Custom modules
# from target_trajectory import TargetTrajectory
# from rewards import RewardDrone

# # --- Safely construct the paths to your assets ---
# ENV_DIR = os.path.dirname(os.path.abspath(__file__))
# Drone_obj_path = os.path.join(ENV_DIR, "Drone_Costum", "Material", "drone_costum.obj")
# TEXTURE_PATH = os.path.join(ENV_DIR, "grass", "textures", "coast_sand_rocks_02_diff_4k.jpg")

# class SuppressPyBulletLog:
#     """Context manager to suppress C-level stdout/stderr from PyBullet."""
#     def __enter__(self):
#         self.devnull = os.open(os.devnull, os.O_WRONLY)
#         self.old_stdout = os.dup(1)
#         self.old_stderr = os.dup(2)
#         os.dup2(self.devnull, 1)
#         os.dup2(self.devnull, 2)

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         os.dup2(self.old_stdout, 1)
#         os.dup2(self.old_stderr, 2)
#         os.close(self.old_stdout)
#         os.close(self.old_stderr)
#         os.close(self.devnull)


# class DroneTrackingEnv(gym.Env):
#     def __init__(self, cfg=None, trajectory_mode: str = "triangular", GUI_mode: bool = False):
#         super().__init__()
        
#         # Enhanced config for Feedforward/Residual Control
#         self.cfg = cfg if cfg is not None else type('obj', (object,), {
#             'seed': 42, 
#             'max_action': 15.0,         # Absolute max velocity of drone (m/s)
#             'max_residual_vel': 2.0,    # Max velocity CORRECTION the RL agent can apply
#             'target_vel_noise': 0.2,    # BoT-SORT velocity estimation noise
#             'max_accel': 10.0,          # Physics acceleration limit
#             'drag': 0.1,                # Air resistance
#             'drone_altitude': 5.0, 
#             'control_period': 0.05, 
#             'sim_dt': 1/240,
#             'world_x': 5000, 'world_y': 5000, 'max_episode_steps': 1000,
#             'drone_start': [0, 0], 'target_start': [0, 0], 'trajectory_scale': 40.0,
#             's_des': 0.05, 'vision': None,
#             'drone_size' : 0.1 
#         })
        
#         self.GUI_mode = GUI_mode
#         self.default_trajectory_mode = trajectory_mode
        
#         # --- Teleop Constraints ---
#         self.teleop_v_limit = 10.0  # Max target speed (m/s)
#         self.teleop_a_limit = 8.0   # Max target acceleration (m/s^2)
        
#         self.rng = np.random.default_rng(self.cfg.seed)

#         # Action: [delta_vx, delta_vy] normalized residual velocity commands [-1, 1]
#         self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
#         # Obs: [x_norm, y_norm, w_norm, h_norm, drone_vx_norm, drone_vy_norm]
#         self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

#         self._client = p.connect(p.GUI) if self.GUI_mode else p.connect(p.DIRECT)
        
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0.0, 0.0, -9.81) 

#         self.plane_id = p.loadURDF("plane.urdf")
        
#         # Load environment texture safely
#         try:
#             texture_id = p.loadTexture(TEXTURE_PATH)
#             p.changeVisualShape(
#                 self.plane_id, -1, 
#                 textureUniqueId=texture_id,
#                 rgbaColor=[1, 1, 1, 1],
#                 specularColor=[0.1, 0.1, 0.1]
#             )
#         except Exception as e:
#             print(f"Warning: Could not load texture at {TEXTURE_PATH}. Falling back to default plane.")
        
#         if GUI_mode:
#             p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
#             p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
#             p.configureDebugVisualizer(lightPosition=[10, 10, 10])
            
#         self.drone_id = self._create_drone_body()
#         self.target_id = self._create_target_body()

#         self.trajectory = None
#         self.current_trajectory_mode = trajectory_mode
#         self.prev_action = np.zeros(2, dtype=np.float32)
        
#         self._compute_reward = RewardDrone()._compute_reward

#     def _create_drone_body(self) -> int:
#         col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
#         visual_orientation = p.getQuaternionFromEuler([math.pi/2, 0 , 0])
#         drone_size = self.cfg.drone_size

#         try:
#             vis = p.createVisualShape(p.GEOM_MESH, fileName=Drone_obj_path, meshScale=[drone_size, drone_size, drone_size],  visualFrameOrientation=visual_orientation)
#         except p.error:
#             vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.1], rgbaColor=[0.1, 0.1, 0.1, 1.0])

#         return p.createMultiBody(
#             baseMass=1.0, 
#             baseCollisionShapeIndex=col, 
#             baseVisualShapeIndex=vis, 
#             basePosition=[self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude]
#         )

#     def _create_target_body(self) -> int:
#         """Loads a Husky rover model from pybullet_data silently."""
#         with SuppressPyBulletLog():
#             target_id = p.loadURDF("husky/husky.urdf", useFixedBase=True)
#         return target_id
    
#     def _update_gui_camera(self):
#         try:
#             cam_info = p.getDebugVisualizerCamera(physicsClientId=self._client)
#             current_yaw = cam_info[8]
#             current_pitch = cam_info[9]
#             current_dist = cam_info[10]
#         except Exception:
#             current_yaw = 45.0
#             current_pitch = -35.0
#             current_dist = 6.0

#         p.resetDebugVisualizerCamera(
#             cameraDistance=current_dist,
#             cameraYaw=current_yaw,
#             cameraPitch=current_pitch,
#             cameraTargetPosition=self.drone_pos.tolist(),
#             physicsClientId=self._client
#         )

#     def _get_obs(self, bbox: Dict[str, float]) -> np.ndarray:
#         vx_norm = np.clip(self.drone_vel[0] / self.cfg.max_action, -1, 1)
#         vy_norm = np.clip(self.drone_vel[1] / self.cfg.max_action, -1, 1)

#         return np.array([
#             bbox.get("x_center", 0.0), 
#             bbox.get("y_center", 0.0), 
#             bbox.get("width", 0.0), 
#             bbox.get("height", 0.0), 
#             vx_norm, 
#             vy_norm
#         ], dtype=np.float32)
    
#     def reset(self, *, seed=None, options=None):
#         super().reset(seed=seed)
#         if seed is not None: 
#             self.rng = np.random.default_rng(seed)
            
#         self.step_count, self.t = 0, 0.0 
#         self.prev_action = np.zeros(2, dtype=np.float32)

#         traj_mode = self.default_trajectory_mode
#         if options is not None and "trajectory_mode" in options:
#             traj_mode = options["trajectory_mode"]
            
#         self.current_trajectory_mode = traj_mode

#         # --- Handle Teleop vs Autonomous Initialization ---
#         if self.current_trajectory_mode == "teleop":
#             self.trajectory = None
#             self.target_pos = np.array([self.cfg.target_start[0], self.cfg.target_start[1], 0.0], dtype=np.float32)
#             self.target_vel = np.zeros(3, dtype=np.float32)
#         else:
#             self.trajectory = TargetTrajectory(
#                 start_xy=self.cfg.target_start, 
#                 scale=self.cfg.trajectory_scale, 
#                 mode=traj_mode
#             )
#             self.target_pos, self.target_vel = self.trajectory.sample(0.0)

#         self.drone_pos = np.array([self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude], dtype=np.float32)
#         self.drone_vel = np.zeros(3, dtype=np.float32)

#         p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
#         p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

#         bbox = self._observe_bbox()
#         info = {"bbox": bbox, "current_trajectory": traj_mode}
        
#         return self._get_obs(bbox), info

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
#         action = np.clip(action, -1.0, 1.0)
        
#         # Simulate BoT-SORT estimation noise
#         estimated_target_vel = self.target_vel[:2] + self.rng.normal(0, self.cfg.target_vel_noise, size=2)
#         residual_vel = action * self.cfg.max_residual_vel
        
#         desired_vel_xy = estimated_target_vel + residual_vel
#         desired_vel_xy = np.clip(desired_vel_xy, -self.cfg.max_action, self.cfg.max_action)

#         self._apply_drone_control(desired_vel_xy)
#         self._advance_target(self.cfg.control_period)
        
#         n_substeps = max(1, int(round(self.cfg.control_period / self.cfg.sim_dt)))
#         for _ in range(n_substeps): p.stepSimulation()
        
#         if self.GUI_mode:
#             self._update_gui_camera()

#         bbox = self._observe_bbox()
#         state = self._get_obs(bbox)

#         target_lost = (abs(state[0]) > 0.98 or abs(state[1]) > 0.98 or bbox["area"] <= 0) 
#         out_of_bounds = not (-self.cfg.world_x < self.drone_pos[0] < self.cfg.world_x 
#                              and -self.cfg.world_y < self.drone_pos[1] < self.cfg.world_y)
        
#         reward, reward_terms = self._compute_reward(state, action, target_lost, self.prev_action)

#         self.step_count += 1
#         self.t += self.cfg.control_period
#         self.prev_action = action.copy()

#         terminated = target_lost
#         truncated = self.step_count >= self.cfg.max_episode_steps or out_of_bounds

#         return state, float(reward), terminated, truncated, {"reward_terms": reward_terms, "bbox": bbox}

#     def _apply_drone_control(self, desired_vel_xy: np.ndarray):
#         accel_cmd = (desired_vel_xy - self.drone_vel[:2]) / self.cfg.control_period
        
#         accel_mag = np.linalg.norm(accel_cmd)
#         if accel_mag > self.cfg.max_accel:
#             accel_cmd *= (self.cfg.max_accel / accel_mag)

#         self.drone_vel[:2] += accel_cmd * self.cfg.control_period
#         self.drone_vel[:2] *= (1.0 - self.cfg.drag * self.cfg.control_period)
        
#         self.drone_pos[:2] += self.drone_vel[:2] * self.cfg.control_period
#         self.drone_pos[2] = self.cfg.drone_altitude 

#         p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
#         p.resetBaseVelocity(self.drone_id, self.drone_vel.tolist())

#     def _observe_bbox(self) -> Dict[str, float]:
#         rel_pos = self.target_pos - self.drone_pos
#         x_img = np.clip(rel_pos[0] / 10.0, -1, 1) 
#         y_img = np.clip(rel_pos[1] / 10.0, -1, 1)
#         area = np.clip(1.0 / (np.linalg.norm(rel_pos) + 1e-5), 0, 1)
#         return {"x_center": x_img, "y_center": y_img, "width": 0.1, "height": 0.1, "area": area}

#     def _advance_target(self, dt: float):
#         # --- TELEOPERATION MODE ---
#         if self.current_trajectory_mode == "teleop":
#             desired_accel = np.zeros(2, dtype=np.float32)
            
#             # Read Keyboard inputs
#             if self.GUI_mode:
#                 keys = p.getKeyboardEvents()
#                 # W = +Y, S = -Y, A = -X, D = +X
#                 if ord('w') in keys and (keys[ord('w')] & p.KEY_IS_DOWN):
#                     desired_accel[1] += self.teleop_a_limit
#                 if ord('s') in keys and (keys[ord('s')] & p.KEY_IS_DOWN):
#                     desired_accel[1] -= self.teleop_a_limit
#                 if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN):
#                     desired_accel[0] -= self.teleop_a_limit
#                 if ord('d') in keys and (keys[ord('d')] & p.KEY_IS_DOWN):
#                     desired_accel[0] += self.teleop_a_limit
            
#             # Apply dynamic braking if no keys are pressed
#             if np.linalg.norm(desired_accel) == 0.0:
#                 brake_accel = -self.target_vel[:2] * 4.0 # Aggressive friction factor
#                 brake_mag = np.linalg.norm(brake_accel)
#                 if brake_mag > self.teleop_a_limit:
#                     brake_accel = (brake_accel / brake_mag) * self.teleop_a_limit
#                 desired_accel = brake_accel
            
#             # Enforce max acceleration limit (diagonal safety)
#             accel_mag = np.linalg.norm(desired_accel)
#             if accel_mag > self.teleop_a_limit:
#                 desired_accel = (desired_accel / accel_mag) * self.teleop_a_limit
                
#             # Kinematic update for Target
#             self.target_vel[:2] += desired_accel * dt
            
#             # Enforce max velocity limit
#             vel_mag = np.linalg.norm(self.target_vel[:2])
#             if vel_mag > self.teleop_v_limit:
#                 self.target_vel[:2] = (self.target_vel[:2] / vel_mag) * self.teleop_v_limit
                
#             self.target_pos[:2] += self.target_vel[:2] * dt
            
#         # --- AUTONOMOUS MODE ---
#         else:
#             self.target_pos, self.target_vel = self.trajectory.sample(self.t + dt)
        
#         # --- UPDATE HUSKY ORIENTATION ---
#         target_speed = np.linalg.norm(self.target_vel[:2])
#         if target_speed > 0.01:
#             yaw = math.atan2(self.target_vel[1], self.target_vel[0])
#         else:
#             yaw = getattr(self, "husky_yaw", 0.0)
#         self.husky_yaw = yaw

#         target_orientation = p.getQuaternionFromEuler([0, 0, yaw])
#         p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), target_orientation)

#     def close(self):
#         if p.isConnected(self._client): p.disconnect(self._client)








import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from typing import Dict, Tuple
import math
import os

# Custom modules
from target_trajectory import TargetTrajectory
from rewards import RewardDrone

# --- Safely construct the paths to your assets ---
ENV_DIR = os.path.dirname(os.path.abspath(__file__))
Drone_obj_path = os.path.join(ENV_DIR, "Drone_Costum", "Material", "drone_costum.obj")
TEXTURE_PATH = os.path.join(ENV_DIR, "grass", "textures", "coast_sand_rocks_02_diff_4k.jpg")

class SuppressPyBulletLog:
    """Context manager to suppress C-level stdout/stderr from PyBullet."""
    def __enter__(self):
        self.devnull = os.open(os.devnull, os.O_WRONLY)
        self.old_stdout = os.dup(1)
        self.old_stderr = os.dup(2)
        os.dup2(self.devnull, 1)
        os.dup2(self.devnull, 2)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stdout, 1)
        os.dup2(self.old_stderr, 2)
        os.close(self.old_stdout)
        os.close(self.old_stderr)
        os.close(self.devnull)


class DroneTrackingEnv(gym.Env):
    def __init__(self, cfg=None, trajectory_mode: str = "triangular", GUI_mode: bool = False):
        super().__init__()
        
        # Enhanced config for Feedforward/Residual Control + Wind
        self.cfg = cfg if cfg is not None else type('obj', (object,), {
            'seed': 42, 
            'max_action': 15.0,         # Absolute max velocity of drone (m/s)
            'max_residual_vel': 2.0,    # Max velocity CORRECTION the RL agent can apply
            'target_vel_noise': 0.2,    # BoT-SORT velocity estimation noise
            'max_accel': 10.0,          # Physics acceleration limit
            'drag': 0.1,                # Air resistance
            'drone_altitude': 5.0, 
            'control_period': 0.05, 
            'sim_dt': 1/240,
            'world_x': 5000, 'world_y': 5000, 'max_episode_steps': 1000,
            'drone_start': [0, 0], 'target_start': [0, 0], 'trajectory_scale': 40.0,
            's_des': 0.05, 'vision': None,
            'drone_size' : 0.1,
            
            # --- WIND CONFIGURATION ---
            'wind_enabled': True,       # Toggle wind on/off
            'max_wind_vel': 6.0,        # Maximum wind speed in m/s
            'wind_volatility': 3.0      # How rapidly the wind changes (m/s^2)
        })
        
        self.GUI_mode = GUI_mode
        self.default_trajectory_mode = trajectory_mode
        
        # --- Teleop Constraints ---
        self.teleop_v_limit = 10.0  # Max target speed (m/s)
        self.teleop_a_limit = 8.0   # Max target acceleration (m/s^2)
        
        self.rng = np.random.default_rng(self.cfg.seed)

        # Action: [delta_vx, delta_vy] normalized residual velocity commands [-1, 1]
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Obs: [x_norm, y_norm, w_norm, h_norm, drone_vx_norm, drone_vy_norm]
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._client = p.connect(p.GUI) if self.GUI_mode else p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81) 

        self.plane_id = p.loadURDF("plane.urdf")
        
        # Load environment texture safely
        try:
            texture_id = p.loadTexture(TEXTURE_PATH)
            p.changeVisualShape(
                self.plane_id, -1, 
                textureUniqueId=texture_id,
                rgbaColor=[1, 1, 1, 1],
                specularColor=[0.1, 0.1, 0.1]
            )
        except Exception as e:
            print(f"Warning: Could not load texture at {TEXTURE_PATH}. Falling back to default plane.")
        
        if GUI_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(lightPosition=[10, 10, 10])
            
        self.drone_id = self._create_drone_body()
        self.target_id = self._create_target_body()

        self.trajectory = None
        self.current_trajectory_mode = trajectory_mode
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.wind_vel = np.zeros(2, dtype=np.float32)
        
        self._compute_reward = RewardDrone()._compute_reward

    def _create_drone_body(self) -> int:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        visual_orientation = p.getQuaternionFromEuler([math.pi/2, 0 , 0])
        drone_size = self.cfg.drone_size

        try:
            vis = p.createVisualShape(p.GEOM_MESH, fileName=Drone_obj_path, meshScale=[drone_size, drone_size, drone_size],  visualFrameOrientation=visual_orientation)
        except p.error:
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.1], rgbaColor=[0.1, 0.1, 0.1, 1.0])

        return p.createMultiBody(
            baseMass=1.0, 
            baseCollisionShapeIndex=col, 
            baseVisualShapeIndex=vis, 
            basePosition=[self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude]
        )

    def _create_target_body(self) -> int:
        """Loads a Husky rover model from pybullet_data silently."""
        with SuppressPyBulletLog():
            target_id = p.loadURDF("husky/husky.urdf", useFixedBase=True)
        return target_id
    
    def _update_gui_camera(self):
        try:
            cam_info = p.getDebugVisualizerCamera(physicsClientId=self._client)
            current_yaw = cam_info[8]
            current_pitch = cam_info[9]
            current_dist = cam_info[10]
        except Exception:
            current_yaw = 45.0
            current_pitch = -35.0
            current_dist = 6.0

        p.resetDebugVisualizerCamera(
            cameraDistance=current_dist,
            cameraYaw=current_yaw,
            cameraPitch=current_pitch,
            cameraTargetPosition=self.drone_pos.tolist(),
            physicsClientId=self._client
        )

    def _get_obs(self, bbox: Dict[str, float]) -> np.ndarray:
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
        super().reset(seed=seed)
        if seed is not None: 
            self.rng = np.random.default_rng(seed)
            
        self.step_count, self.t = 0, 0.0 
        self.prev_action = np.zeros(2, dtype=np.float32)

        # --- INITIALIZE WIND ---
        if getattr(self.cfg, 'wind_enabled', False):
            self.wind_vel = self.rng.uniform(-self.cfg.max_wind_vel, self.cfg.max_wind_vel, size=2).astype(np.float32)
        else:
            self.wind_vel = np.zeros(2, dtype=np.float32)

        traj_mode = self.default_trajectory_mode
        if options is not None and "trajectory_mode" in options:
            traj_mode = options["trajectory_mode"]
            
        self.current_trajectory_mode = traj_mode

        # --- Handle Teleop vs Autonomous Initialization ---
        if self.current_trajectory_mode == "teleop":
            self.trajectory = None
            self.target_pos = np.array([self.cfg.target_start[0], self.cfg.target_start[1], 0.0], dtype=np.float32)
            self.target_vel = np.zeros(3, dtype=np.float32)
        else:
            self.trajectory = TargetTrajectory(
                start_xy=self.cfg.target_start, 
                scale=self.cfg.trajectory_scale, 
                mode=traj_mode
            )
            self.target_pos, self.target_vel = self.trajectory.sample(0.0)

        self.drone_pos = np.array([self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude], dtype=np.float32)
        self.drone_vel = np.zeros(3, dtype=np.float32)

        p.resetBasePositionAndOrientation(self.drone_id, self.drone_pos.tolist(), [0,0,0,1])
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), [0,0,0,1])

        bbox = self._observe_bbox()
        # Add wind to info dict so it can be logged during training
        info = {"bbox": bbox, "current_trajectory": traj_mode, "wind_vel": self.wind_vel.copy()}
        
        return self._get_obs(bbox), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0)
        
        # --- SIMULATE SHIFTING WIND GUSTS ---
        if getattr(self.cfg, 'wind_enabled', False):
            # Smoothly shift the wind vector using normal noise
            wind_change = self.rng.normal(0, self.cfg.wind_volatility, size=2) * self.cfg.control_period
            self.wind_vel += wind_change
            
            # Cap the wind speed to the maximum allowed limit
            wind_speed = np.linalg.norm(self.wind_vel)
            if wind_speed > self.cfg.max_wind_vel:
                self.wind_vel = (self.wind_vel / wind_speed) * self.cfg.max_wind_vel
        
        # Simulate BoT-SORT estimation noise
        estimated_target_vel = self.target_vel[:2] + self.rng.normal(0, self.cfg.target_vel_noise, size=2)
        residual_vel = action * self.cfg.max_residual_vel
        
        desired_vel_xy = estimated_target_vel + residual_vel
        desired_vel_xy = np.clip(desired_vel_xy, -self.cfg.max_action, self.cfg.max_action)

        self._apply_drone_control(desired_vel_xy)
        self._advance_target(self.cfg.control_period)
        
        n_substeps = max(1, int(round(self.cfg.control_period / self.cfg.sim_dt)))
        for _ in range(n_substeps): p.stepSimulation()
        
        if self.GUI_mode:
            self._update_gui_camera()

        bbox = self._observe_bbox()
        state = self._get_obs(bbox)

        target_lost = (abs(state[0]) > 0.98 or abs(state[1]) > 0.98 or bbox["area"] <= 0) 
        out_of_bounds = not (-self.cfg.world_x < self.drone_pos[0] < self.cfg.world_x 
                             and -self.cfg.world_y < self.drone_pos[1] < self.cfg.world_y)
        
        reward, reward_terms = self._compute_reward(state, action, target_lost, self.prev_action)

        self.step_count += 1
        self.t += self.cfg.control_period
        self.prev_action = action.copy()

        terminated = target_lost
        truncated = self.step_count >= self.cfg.max_episode_steps or out_of_bounds

        # Pass current wind vel in info dict
        return state, float(reward), terminated, truncated, {"reward_terms": reward_terms, "bbox": bbox, "wind_vel": self.wind_vel.copy()}

    def _apply_drone_control(self, desired_vel_xy: np.ndarray):
        # Calculate acceleration command
        accel_cmd = (desired_vel_xy - self.drone_vel[:2]) / self.cfg.control_period
        
        # Enforce max acceleration
        accel_mag = np.linalg.norm(accel_cmd)
        if accel_mag > self.cfg.max_accel:
            accel_cmd *= (self.cfg.max_accel / accel_mag)

        # Apply control acceleration
        self.drone_vel[:2] += accel_cmd * self.cfg.control_period
        
        # --- APPLY WIND & DRAG ---
        # Calculate velocity relative to the moving air mass (wind)
        relative_vel = self.drone_vel[:2] - self.wind_vel
        self.drone_vel[:2] -= relative_vel * (self.cfg.drag * self.cfg.control_period)
        
        # Update spatial position based on final velocities
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
        # --- TELEOPERATION MODE ---
        if self.current_trajectory_mode == "teleop":
            desired_accel = np.zeros(2, dtype=np.float32)
            
            # Read Keyboard inputs
            if self.GUI_mode:
                keys = p.getKeyboardEvents()
                # W = +Y, S = -Y, A = -X, D = +X
                if ord('w') in keys and (keys[ord('w')] & p.KEY_IS_DOWN):
                    desired_accel[1] += self.teleop_a_limit
                if ord('s') in keys and (keys[ord('s')] & p.KEY_IS_DOWN):
                    desired_accel[1] -= self.teleop_a_limit
                if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN):
                    desired_accel[0] -= self.teleop_a_limit
                if ord('d') in keys and (keys[ord('d')] & p.KEY_IS_DOWN):
                    desired_accel[0] += self.teleop_a_limit
            
            # Apply dynamic braking if no keys are pressed
            if np.linalg.norm(desired_accel) == 0.0:
                brake_accel = -self.target_vel[:2] * 4.0 # Aggressive friction factor
                brake_mag = np.linalg.norm(brake_accel)
                if brake_mag > self.teleop_a_limit:
                    brake_accel = (brake_accel / brake_mag) * self.teleop_a_limit
                desired_accel = brake_accel
            
            # Enforce max acceleration limit (diagonal safety)
            accel_mag = np.linalg.norm(desired_accel)
            if accel_mag > self.teleop_a_limit:
                desired_accel = (desired_accel / accel_mag) * self.teleop_a_limit
                
            # Kinematic update for Target
            self.target_vel[:2] += desired_accel * dt
            
            # Enforce max velocity limit
            vel_mag = np.linalg.norm(self.target_vel[:2])
            if vel_mag > self.teleop_v_limit:
                self.target_vel[:2] = (self.target_vel[:2] / vel_mag) * self.teleop_v_limit
                
            self.target_pos[:2] += self.target_vel[:2] * dt
            
        # --- AUTONOMOUS MODE ---
        else:
            self.target_pos, self.target_vel = self.trajectory.sample(self.t + dt)
        
        # --- UPDATE HUSKY ORIENTATION ---
        target_speed = np.linalg.norm(self.target_vel[:2])
        if target_speed > 0.01:
            yaw = math.atan2(self.target_vel[1], self.target_vel[0])
        else:
            yaw = getattr(self, "husky_yaw", 0.0)
        self.husky_yaw = yaw

        target_orientation = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.target_id, self.target_pos.tolist(), target_orientation)

    def close(self):
        if p.isConnected(self._client): p.disconnect(self._client)