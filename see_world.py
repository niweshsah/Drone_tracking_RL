import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --- CONFIGURATION ---

@dataclass
class EnvConfig:
    seed: int = 42
    control_period: float = 0.1      
    sim_dt: float = 1.0 / 240.0      
    max_episode_steps: int = 1000     
    world_x: float = 500.0           # Increased logical bounds
    world_y: float = 500.0
    drone_altitude: float = 10.0      
    target_altitude: float = 0.0     
    max_action: float = 6.0          
    max_accel: float = 4.0           
    drag: float = 0.1              
    x_des: float = 0.5               
    s_des: float = 0.06              
    w1, w2, w3 = 1.0, 0.15, 0.15

# --- MOCK VISION ---

class VisionBBoxEstimator:
    def __init__(self, cfg=None, rng=None): pass
    def reset(self): pass
    def project_bbox(self, drone_pos, yaw, target_pos, occluded=False):
        return {"x_center": 0.5, "area": 0.06, "occluded": occluded}

# --- TRAJECTORY LOGIC ---

class TargetTrajectory:
    def __init__(self, mode: str):
        self.mode = mode
        self.segment_duration = 5.0   
        self.pause_duration = 0.5     
        # Defined waypoints for the path
        points = {
            "triangular": [[30, 100, 0], [150, 250, 0], [270, 100, 0], [30, 100, 0]],
            "square_wave": [[30, 100, 0], [100, 100, 0], [100, 200, 0], [200, 200, 0], [200, 100, 0], [300, 100, 0]]
        }.get(mode, [[30,100,0], [300,300,0]])
        
        self.points = [np.asarray(pt, dtype=np.float32) for pt in points]
        self.n_segments = len(self.points) - 1

    def duration(self) -> float:
        return self.n_segments * (self.segment_duration + self.pause_duration)

    def sample(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        cycle = self.duration()
        t_mod = t % cycle
        seg_idx = min(int(t_mod // (self.segment_duration + self.pause_duration)), self.n_segments - 1)
        t_local = min(t_mod % (self.segment_duration + self.pause_duration), self.segment_duration)
        
        p0, p1 = self.points[seg_idx], self.points[seg_idx + 1]
        alpha = t_local / self.segment_duration
        pos = p0 + alpha * (p1 - p0)
        vel = (p1 - p0) / self.segment_duration
        return pos, vel

# --- ENVIRONMENT ---

class DroneTrackingEnv(gym.Env):
    def __init__(self, trajectory_mode: str = "square_wave"):
        super().__init__()
        self.cfg = EnvConfig()
        
        # Connect to GUI
        self._client = p.connect(p.GUI) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # --- THE FIX: Load the plane with globalScaling=20 ---
        # This makes the floor 20 times larger than the default.
        self.plane_id = p.loadURDF("plane.urdf", basePosition=[0,0,0], globalScaling=20)
        
        # Drone (Blue) and Target (Red)
        self.drone_id = self._create_box([1.0, 1.0, 0.3], [0.2, 0.5, 0.9, 1]) # Slightly bigger drone for visibility
        self.target_id = self._create_box([2.0, 2.0, 0.8], [0.9, 0.2, 0.2, 1]) # Bigger target
        
        self.trajectory = TargetTrajectory(trajectory_mode)

    def _create_box(self, size, color):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s/2 for s in size])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s/2 for s in size], rgbaColor=color)
        return p.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis)

    def reset(self, seed=None):
        self.t = 0.0
        return np.zeros(2), {}

    def step(self, action):
        self.t += self.cfg.control_period
        p.stepSimulation()
        return np.zeros(2), 0.0, False, False, {}

    def close(self):
        p.disconnect()

# --- THE VISUALIZER ---

def visualize_ideal_tracking():
    # Try "square_wave" or "triangular"
    env = DroneTrackingEnv(trajectory_mode="triangular")
    env.reset()
    
    # Remove the side panels for a cleaner view
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    print("\n--- Visualizing Ideal World (Large Floor) ---")

    try:
        while True:
            # 1. Get positions
            target_pos, _ = env.trajectory.sample(env.t)
            drone_pos = [target_pos[0], target_pos[1], env.cfg.drone_altitude]
            
            # 2. Update positions in Bullet
            p.resetBasePositionAndOrientation(env.target_id, target_pos, [0,0,0,1])
            p.resetBasePositionAndOrientation(env.drone_id, drone_pos, [0,0,0,1])
            
            # 3. Dynamic Follow Camera
            p.resetDebugVisualizerCamera(
                cameraDistance=40.0, # Zoomed out a bit more to see the path
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=drone_pos
            )
            
            # 4. Visual "Tether" line (Green)
            p.addUserDebugLine(target_pos, drone_pos, [0, 1, 0], lineWidth=3, lifeTime=env.cfg.control_period)

            # 5. Optional: Leave a "breadcrumb" path for the target (Red)
            p.addUserDebugLine(target_pos, target_pos + [0,0,0.1], [1, 0, 0], lineWidth=2, lifeTime=10.0)

            env.step(None)
            time.sleep(env.cfg.control_period)

    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    visualize_ideal_tracking()