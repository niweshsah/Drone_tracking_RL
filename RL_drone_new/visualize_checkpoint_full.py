import argparse
import csv
import logging
import os
import time
from typing import Dict, List

import numpy as np
import pybullet as p

from environment import DroneTrackingEnv
from agent import TD3Agent, TD3Config
from utils import compute_tracking_metrics

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def evaluate_checkpoint(
    checkpoint_path: str, 
    trajectories: List[str], 
    episodes_per_traj: int = 3, 
    render: bool = True
):
    logging.info("="*60)
    logging.info(f"EVALUATING CHECKPOINT: {checkpoint_path}")
    logging.info("="*60)

    # 1. Initialize Environment
    env = DroneTrackingEnv(cfg=None, trajectory_mode="triangular", GUI_mode=render)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = 1.0

    # 2. Initialize Agent & Load Weights
    td3_cfg = TD3Config(
        state_dim=state_dim, action_dim=action_dim, max_action=max_action, 
        hidden_dim=256, device="cpu"
    )
    agent = TD3Agent(td3_cfg)

    try:
        agent.load(checkpoint_path, strict=False)
        agent.actor.eval()
        logging.info("✔ Weights loaded successfully.\n")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        env.close()
        return

    # Prepare data storage
    all_results = []
    
    try:
        # 3. Iterate through each specified trajectory
        for traj in trajectories:
            logging.info(f"--- Testing Trajectory: {traj.upper()} ---")
            
            traj_rewards = []
            traj_metrics = {"x_err": [], "y_err": [], "jitter": [], "jerk": []}

            for ep in range(episodes_per_traj):
                # Using the options dictionary we built for the curriculum!
                obs, _ = env.reset(options={"trajectory_mode": traj})
                ep_reward = 0.0
                
                # Kinematic history arrays for metric calculation
                drone_xy_hist, target_xy_hist, drone_vel_hist = [], [], []

                if render:
                    for _ in range(10): p.stepSimulation() # Settle physics
                
                done, truncated = False, False
                
                while not (done or truncated):
                    action = agent.select_action(obs)
                    next_obs, reward, done, truncated, _ = env.step(action)
                    
                    # Record kinematics
                    drone_xy_hist.append(env.drone_pos[:2].copy())
                    target_xy_hist.append(env.target_pos[:2].copy())
                    drone_vel_hist.append(env.drone_vel[:2].copy())
                    
                    ep_reward += reward
                    obs = next_obs
                    
                    # Rendering Logic
                    if render:
                        p.addUserDebugLine(
                            env.target_pos, env.drone_pos, [0, 1, 0], 
                            lineWidth=2.5, lifeTime=env.cfg.control_period
                        )
                        time.sleep(env.cfg.control_period)

                # End of Episode: Compute Metrics
                metrics = compute_tracking_metrics(
                    np.array(drone_xy_hist), 
                    np.array(target_xy_hist), 
                    np.array(drone_vel_hist), 
                    env.cfg.control_period
                )
                
                traj_rewards.append(ep_reward)
                traj_metrics["x_err"].append(metrics["x_tracking_error"])
                traj_metrics["y_err"].append(metrics["y_tracking_error"])
                traj_metrics["jitter"].append(metrics["velocity_jitter"])
                traj_metrics["jerk"].append(metrics["jerk_rms"])
                
                logging.info(f"  Ep {ep+1}/{episodes_per_traj} | Reward: {ep_reward:7.1f} | X-Err: {metrics['x_tracking_error']:5.2f}m | Y-Err: {metrics['y_tracking_error']:5.2f}m")

            # Aggregate averages for this trajectory
            avg_result = {
                "trajectory": traj,
                "avg_reward": np.mean(traj_rewards),
                "avg_x_err": np.mean(traj_metrics["x_err"]),
                "avg_y_err": np.mean(traj_metrics["y_err"]),
                "avg_jitter": np.mean(traj_metrics["jitter"]),
                "avg_jerk_rms": np.mean(traj_metrics["jerk"])
            }
            all_results.append(avg_result)
            logging.info("") # Empty line for readability

    except KeyboardInterrupt:
        logging.info("\nEvaluation manually terminated. Saving collected data...")
    
    finally:
        env.close()
        
    # 4. Save and Print Results
    save_and_print_results(all_results, checkpoint_path)


def save_and_print_results(results: List[Dict], checkpoint_path: str):
    if not results:
        return
        
    # Determine save path (save in the same folder as the checkpoint)
    ckpt_dir = os.path.dirname(checkpoint_path)
    csv_path = os.path.join(ckpt_dir, "evaluation_metrics.csv")
    
    # Write to CSV
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    # Print Summary Table
    print("\n" + "="*85)
    print(f"{'TRAJECTORY':<15} | {'REWARD':<10} | {'X-ERR (m)':<10} | {'Y-ERR (m)':<10} | {'JITTER':<10} | {'JERK RMS':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['trajectory']:<15} | {r['avg_reward']:<10.1f} | {r['avg_x_err']:<10.3f} | {r['avg_y_err']:<10.3f} | {r['avg_jitter']:<10.3f} | {r['avg_jerk_rms']:<10.2f}")
    print("="*85)
    print(f"Metrics saved to: {csv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TD3 Drone Tracker")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt file")
    parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI (much faster)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes per trajectory")
    
    args = parser.parse_args()
    
    # Test all available trajectories
    TEST_TRAJECTORIES = ["triangular", "square", "sawtooth", "square_wave"]
    
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint, 
        trajectories=TEST_TRAJECTORIES, 
        episodes_per_traj=args.episodes, 
        render=not args.headless # Invert headless to get render flag
    )