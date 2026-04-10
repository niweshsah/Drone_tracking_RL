import torch
import numpy as np
import pybullet as p
from environment import DroneTrackingEnv, EnvConfig
from live_view import Actor

def debug_agent(checkpoint_path):
    # 1. Setup
    cfg = EnvConfig(max_action=6.0, control_period=0.3)
    env = DroneTrackingEnv(cfg, trajectory_mode="triangular")
    
    # Load Actor
    actor = Actor(max_action=cfg.max_action)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["actor"] if "actor" in ckpt else ckpt
    actor.load_state_dict(state_dict)
    actor.eval()

    print(f"\n{'='*80}")
    print(f"{'STEP':<6} | {'DIST (m)':<10} | {'BBOX (x, area)':<15} | {'STATE (s1, s2)':<15} | {'ACTION (fb, lr)':<15}")
    print(f"{'='*80}")

    state, info = env.reset()
    
    try:
        for i in range(100):
            # Capture raw physical distance for debugging
            real_dist_x = env.drone_pos[0] - env.target_pos[0]
            real_dist_y = env.drone_pos[1] - env.target_pos[1]
            total_dist = np.sqrt(real_dist_x**2 + real_dist_y**2)

            # Get Actor Output
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_t).numpy()[0]
            
            # Extract vision data from info (if your env provides it)
            bbox = info.get("bbox", {"x_center": 0.0, "area": 0.0})
            
            # Print Every 10 steps or if distance gets huge
            if i % 10 == 0 or total_dist > 150:
                print(f"{i:<6} | {total_dist:<10.2f} | "
                      f"({bbox['x_center']:>4.2f}, {bbox['area']:>6.4f}) | "
                      f"({state[0]:>5.2f}, {state[1]:>5.2f}) | "
                      f"({action[0]:>5.2f}, {action[1]:>5.2f})")

            # Check for critical failures
            if bbox['area'] == 0:
                print(f"!!! CRITICAL: Target LOST at step {i}. Check camera orientation.")
            
            state, reward, done, truncated, info = env.step(action)
            
            if total_dist > 200:
                print("!!! FAILURE: Drone is essentially in space. Stopping debug.")
                break
                
    finally:
        env.close()

if __name__ == "__main__":
    debug_agent("/home/rocinate/Desktop/DL-workspace-pytorch/RL_drone/checkpoints_dslab/best.pt")