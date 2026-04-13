import gymnasium as gym
import numpy as np
import pybullet as p
from typing import Dict, Tuple, Optional

# gym-pybullet-drones imports
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# Assuming EnvConfig, TargetTrajectory, and VisionBBoxEstimator 
# are defined exactly as in your previous file.
from your_config_file import EnvConfig, TargetTrajectory, VisionBBoxEstimator, vision_state_from_bbox


class VisionTrackingAviary(BaseRLAviary):
    """
    True quadcopter simulation for Vision-Based Tracking.
    Inherits from BaseRLAviary to utilize accurate aerodynamic models.
    """
    def __init__(self, 
                 cfg: Optional[EnvConfig] = None, 
                 trajectory_mode: str = "triangular",
                 drone_model: DroneModel = DroneModel.CF2X,
                 physics: Physics = Physics.PYB,
                 gui=False):
        
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        
        # RL Control frequency vs Physics frequency
        self.CTRL_FREQ = int(1.0 / self.cfg.control_period)
        self.PYB_FREQ = int(1.0 / self.cfg.sim_dt)
        
        # Initialize the Target Trajectory
        self.trajectory = TargetTrajectory(
            trajectory_mode, 
            self.cfg.target_start, 
            self.cfg.trajectory_scale
        )
        self.vision = VisionBBoxEstimator(self.cfg.vision, self.rng)
        
        # State tracking variables
        self.t = 0.0
        self.prev_state = np.zeros(2, dtype=np.float32)
        self.target_id = None
        
        # Initialize the BaseRLAviary
        super().__init__(
            drone_model=drone_model,
            num_drones=1,
            neighbourhood_radius=10,
            initial_xyzs=np.array([[self.cfg.drone_start[0], self.cfg.drone_start[1], self.cfg.drone_altitude]]),
            initial_rpys=np.array([[0, 0, 0]]),
            physics=physics,
            pyb_freq=self.PYB_FREQ,
            ctrl_freq=self.CTRL_FREQ,
            gui=gui,
            record=False,
            obs=None, # Overridden below
            act=None  # Overridden below
        )

        # Initialize the Low-Level Flight Controller (PID)
        # This handles the complex translation from Velocity -> Motor RPMs
        self.ctrl = DSLPIDControl(drone_model=drone_model)

    # --- REQUIRED GYM-PYBULLET-DRONES OVERRIDES ---

    def _actionSpace(self):
        """RL Agent outputs normalized [v_x, v_y] in range [-1, 1]."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _observationSpace(self):
        """Obs: [x_error, size_error] normalized relative to target."""
        return gym.spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)

    def _preprocessAction(self, action: np.ndarray) -> np.ndarray:
        """
        Transforms the RL agent's normalized velocity command into 4 motor RPMs.
        This is the core of the Hierarchical Control split.
        """
        # 1. Denormalize action to physical velocity limits
        action = np.clip(action, -1.0, 1.0)
        v_x = action[0] * self.cfg.max_action
        v_y = action[1] * self.cfg.max_action
        v_z = 0.0 # Drone must not command vertical velocity
        
        # 2. Get current physical state of the drone
        state = self._getDroneStateVector(0)
        cur_pos = state[0:3]
        
        # 3. Create a target setpoint for the PID using Euler integration
        # We want the drone to move at (v_x, v_y), so we project a target position slightly ahead
        target_pos = cur_pos + np.array([v_x, v_y, v_z]) * self.CTRL_TIMESTEP
        target_pos[2] = self.cfg.drone_altitude # Strictly enforce altitude lock
        
        # 4. Compute required motor RPMs using the built-in PID
        rpm, _, _ = self.ctrl.computeControlFromKinematics(
            control_timestep=self.CTRL_TIMESTEP,
            state=state,
            target_pos=target_pos,
            target_vel=np.array([v_x, v_y, v_z])
        )
        
        # Return RPMs formatted for a single drone: shape (1, 4)
        return np.array([rpm])

    def _computeObs(self) -> np.ndarray:
        """Extracts the simulated optical bounding box state."""
        state_vec = self._getDroneStateVector(0)
        drone_pos = state_vec[0:3]
        
        # Ensure target pos is updated to the current time t
        target_pos, _ = self.trajectory.sample(self.t)
        
        # Simulate Vision
        bbox = self.vision.project_bbox(drone_pos, 0.0, target_pos, occluded=False)
        state = vision_state_from_bbox(bbox["x_center"], bbox["area"], self.cfg.x_des, self.cfg.s_des)
        
        return state.astype(np.float32)

    def _computeReward(self) -> float:
        """Computes the VTD3 composite reward."""
        s_t = self._computeObs()
        s1_curr, s2_curr = s_t[0], s_t[1]
        s1_prev, s2_prev = self.prev_state[0], self.prev_state[1]
        
        # Retrieve the last taken action (stored by the base class)
        last_action = self.action_buffer[-1][0] if len(self.action_buffer) > 0 else np.zeros(2)
        fb, lr = last_action[0], last_action[1]

        # Rs: Distance/Alignment Reward
        rs = (abs(s1_prev) - abs(s1_curr)) + (abs(s2_prev) - abs(s2_curr))

        # Rspeed: High speed ONLY if moving TOWARD the target
        rspeed = -10.0
        far_cond = (s2_curr < -1.5) and (fb > 0.9) # Note: Action is now normalized [-1, 1]
        close_cond = (s2_curr > 1.5) and (fb < -0.9)
        if far_cond or close_cond:
            rspeed = 1.0

        # Rstability: Precision hovering near target
        rstability = -10.0
        if abs(s1_curr) < 0.015 and abs(s2_curr) < 0.02:
            if abs(fb) < 0.1 and abs(lr) < 0.1:
                rstability = 1.0

        # Update previous state for next step
        self.prev_state = s_t.copy()

        return float((self.cfg.w1 * rs) + (self.cfg.w2 * rspeed) + (self.cfg.w3 * rstability))

    def _computeTerminated(self) -> bool:
        """Episode finishes if target is tracked successfully for max steps."""
        return False

    def _computeTruncated(self) -> bool:
        """Episode truncates if the drone flies out of bounds or crashes."""
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        
        out_of_bounds = not (-self.cfg.world_x < pos[0] < self.cfg.world_x and 
                             -self.cfg.world_y < pos[1] < self.cfg.world_y)
        
        crashed = pos[2] < 0.1 # Altitude dropped too low (PID failure or extreme bank angle)
        
        timeout = self.step_counter / self.PYB_FREQ >= self.cfg.max_episode_steps * self.cfg.control_period
        
        return out_of_bounds or crashed or timeout

    def _computeInfo(self) -> dict:
        return {}

    # --- ENVIRONMENT SETUP OVERRIDES ---

    def _addObstacles(self):
        """Adds the moving target to the PyBullet physics engine."""
        super()._addObstacles() # Loads the ground plane
        
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.8])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.8, 0.8, 0.8], rgbaColor=[0.9, 0.2, 0.2, 1.0])
        
        start_pos, _ = self.trajectory.sample(0.0)
        self.target_id = p.createMultiBody(
            baseMass=0.0, # Mass 0 makes it kinematic (unaffected by gravity/drone downwash)
            baseCollisionShapeIndex=col, 
            baseVisualShapeIndex=vis,
            basePosition=start_pos.tolist()
        )

    def step(self, action):
        """Overrides the standard step to advance the target's trajectory."""
        # 1. Advance the simulation time
        self.t += self.CTRL_TIMESTEP
        
        # 2. Move the target object in PyBullet manually
        target_pos, _ = self.trajectory.sample(self.t)
        p.resetBasePositionAndOrientation(
            self.target_id, 
            target_pos.tolist(), 
            [0,0,0,1], 
            physicsClientId=self.CLIENT
        )
        
        # 3. Proceed with normal RL steps (action -> PID -> physics -> reward)
        return super().step(action)
    
    def reset(self, seed=None, options=None):
        """Resets the environment and the target clock."""
        self.t = 0.0
        self.prev_state = np.zeros(2, dtype=np.float32)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        obs, info = super().reset(seed=seed, options=options)
        self.prev_state = obs.copy()
        return obs, info