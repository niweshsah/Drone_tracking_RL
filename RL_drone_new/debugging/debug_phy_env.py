import numpy as np
import time
from physics_env import DroneTrackingEnv 

def debug_trajectory_tracking(mode="circular"):
    print(f"=== Starting DroneTrackingEnv Debug: {mode.upper()} Trajectory ===\n")

    # Force GUI_mode to False if you are on a headless server
    env = DroneTrackingEnv(GUI_mode=False, trajectory_mode=mode)
    
    try:
        print(f"--- Initializing {mode} Episode ---")
        obs, info = env.reset(seed=42)
        
        # Track tracking error (distance from center of image)
        tracking_errors = []
        
        print(f"Running {env.cfg.max_episode_steps} steps of autonomous tracking...")
        
        for i in range(env.cfg.max_episode_steps):
            # Test Case: Zero residual action. 
            # If your feedforward logic is correct, the drone should follow the 
            # target perfectly (minus the BoT-SORT noise).
            action = np.array([0.0, 0.0], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # obs[0] = x_center, obs[1] = y_center
            dist_from_center = np.linalg.norm(obs[:2])
            tracking_errors.append(dist_from_center)
            
            if i % 100 == 0:
                print(f"Step {i:4d} | Error: {dist_from_center:.4f} | Reward: {reward:.2f}")
                # print(f"      Reward Terms: {info.get('reward_terms')}")

            if terminated:
                print(f"❌ Target LOST at step {i}!")
                break
            
            if truncated:
                print(f"✅ Episode finished successfully (max steps).")
                break

        # Post-test analysis
        avg_error = np.mean(tracking_errors)
        max_error = np.max(tracking_errors)
        print("\n--- Tracking Analysis ---")
        print(f"Average Tracking Error (0 is perfect): {avg_error:.4f}")
        print(f"Maximum Tracking Error: {max_error:.4f}")
        
        if avg_error < 0.2:
            print("💎 STATUS: Feedforward logic is EXCELLENT.")
        elif avg_error < 0.5:
            print("⚠️ STATUS: Tracking is stable but loose. Check noise levels.")
        else:
            print("🔥 STATUS: Tracking is failing. Drone is drifting away.")

    except Exception as e:
        print(f"❌ DEBUG FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print("\n=== Debug Session Finished ===")

if __name__ == "__main__":
    # You can change this to "triangular", "circular", or "straight"
    debug_trajectory_tracking(mode="triangular")