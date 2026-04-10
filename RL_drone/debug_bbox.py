import numpy as np
import pybullet as p
from environment import DroneTrackingEnv, EnvConfig

def run_bbox_validation():
    # 1. Setup Config with ZERO noise for perfectly clean data
    cfg = EnvConfig()
    cfg.vision.detection_noise_std_x = 0.0
    cfg.vision.detection_noise_std_area = 0.0
    cfg.vision.dropout_prob = 0.0
    
    # 2. Initialize Environment
    env = DroneTrackingEnv(cfg)
    
    # FORCE STATIONARY TARGET: Overwrite the trajectory with a fixed point
    # Target stays at [30, 100, 0] forever
    fixed_target = np.array([30.0, 100.0, 0.0])
    env.trajectory.sample = lambda t: (fixed_target, np.zeros(3))
    
    print(f"\n{'='*85}")
    print(f"{'STEP':<6} | {'DIST (m)':<10} | {'BBOX AREA':<12} | {'S2 STATE':<10} | {'TREND':<15}")
    print(f"{'='*85}")

    state, info = env.reset()
    prev_dist = 30.0
    prev_area = 0.0
    
    # We will fly forward at 2.0 m/s for 30 steps
    # At 0.3s per step, we should cover 18 meters (Distance 30 -> 12)
    for i in range(30):
        # Force a constant forward velocity action
        action = np.array([2.0, 0.0]) 
        
        # Step environment
        state, reward, done, truncated, info = env.step(action)
        
        # Calculate values
        curr_dist = np.linalg.norm(env.target_pos - env.drone_pos)
        bbox = info.get("bbox", {"area": 0.0})
        curr_area = bbox['area']
        s2_val = state[1] # Distance error state
        
        # Determine Trends
        dist_trend = "DOWN ✅" if curr_dist < prev_dist else "UP ❌"
        area_trend = "UP ✅" if curr_area > prev_area else "DOWN ❌"
        
        print(f"{i:<6} | {curr_dist:<10.2f} | {curr_area:<12.6f} | {s2_val:<10.4f} | {dist_trend} / {area_trend}")

        prev_dist = curr_dist
        prev_area = curr_area
        
        if curr_dist < 1.0:
            print("\n--- Drone has reached the target! ---")
            break

    env.close()

if __name__ == "__main__":
    run_bbox_validation()