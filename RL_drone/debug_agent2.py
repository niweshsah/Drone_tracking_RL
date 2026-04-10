import torch
import numpy as np
import pybullet as p
import logging
from environment import DroneTrackingEnv, EnvConfig
from actor_fwd import Actor


def setup_logger(log_path="debug_log_agent.log"):
    logger = logging.getLogger("debug_logger")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)

    # Format
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    return logger


def debug_physics_test(checkpoint_path):
    cfg = EnvConfig(max_action=6.0, control_period=0.3)
    env = DroneTrackingEnv(cfg, trajectory_mode="triangular")
    
    # We don't even need to load the actor for this test!
    print(f"\n{'='*60}")
    print(f"{'STEP':<6} | {'DIST (m)':<10} | {'BBOX (x, area)':<15} | {'ACTION':<10}")
    print(f"{'='*60}")

    state, info = env.reset()
    
    for i in range(20): # 20 steps is enough to see the trend
        # --- THE HARDCODE PART ---
        # Force the drone to fly FORWARD (Positive X)
        action = np.array([1.0, 0.0]) 
        
        # Calculate real distance
        real_dist_x = env.target_pos[0] - env.drone_pos[0]
        real_dist_y = env.target_pos[1] - env.drone_pos[1]
        total_dist = np.sqrt(real_dist_x**2 + real_dist_y**2)

        bbox = info.get("bbox", {"x_center": 69.0, "area": 69.0})
        
        print(f"{i:<6} | {total_dist:<10.2f} | ({bbox['x_center']:>4.2f}, {bbox['area']:>6.4f}) | {action}")

        state, reward, done, truncated, info = env.step(action)
    
    env.close()
    
def debug_physics_static():
    cfg = EnvConfig()
    env = DroneTrackingEnv(cfg)
    
    # MANUALLY STOP THE TARGET
    env.trajectory.points = [np.array([30, 100, 0]), np.array([30, 100, 0])] 
    
    state, info = env.reset()
    print(f"\n{'STEP':<6} | {'DIST':<10} | {'BBOX (x, area)':<15} | {'STATE':<15}")
    
    for i in range(20):
        action = np.array([2.0, 0.0]) # Fly faster (2m/s) at a stationary target
        state, reward, done, truncated, info = env.step(action)
        
        dist = np.linalg.norm(env.target_pos - env.drone_pos)
        bbox = info.get("bbox", {"x_center": 69.0, "area": 69.0})
        
        print(f"{i:<6} | {dist:<10.2f} | ({bbox['x_center']:>4.2f}, {bbox['area']:>6.4f}) | ({state[0]:>5.2f}, {state[1]:>5.2f})")
    env.close()

if __name__ == "__main__":
    # debug_physics_static() # No checkpoint needed for static test
    debug_physics_test(None) # No checkpoint needed for physics test