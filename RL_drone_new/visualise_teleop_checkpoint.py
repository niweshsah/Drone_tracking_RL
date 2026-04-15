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


def setup_logger(checkpoint_path: str) -> str:
    """Configures the logger to write to both the console and a log file."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    if not ckpt_dir:
        ckpt_dir = "."
    log_file = os.path.join(ckpt_dir, "evaluation.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(message)s')

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return log_file


def evaluate_checkpoint(
    checkpoint_path: str, 
    trajectories: List[str], 
    episodes_per_traj: int = 1, 
    render: bool = True
):
    # Setup dual-logging before doing anything else
    log_file = setup_logger(checkpoint_path)
    
    logging.info("="*60)
    logging.info(f"EVALUATING CHECKPOINT: {checkpoint_path}")
    logging.info(f"LOG FILE SAVED TO: {log_file}")
    logging.info("="*60)

    # 1. Initialize Environment
    # If teleop is in the list, we MUST render the GUI to capture keyboard inputs
    if "teleop" in trajectories and not render:
        logging.warning("⚠️ 'teleop' requires keyboard input. Overriding --headless and forcing GUI render!")
        render = True

    env = DroneTrackingEnv(cfg=None, trajectory_mode="square", GUI_mode=render)
    
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
            
            if traj == "teleop":
                logging.info("\n🕹️  TELEOP MODE ACTIVE! 🕹️")
                logging.info("-> Click the PyBullet window to focus it.")
                logging.info("-> Use W, A, S, D to drive the Husky.")
                logging.info("-> Press CTRL+C in the terminal to end early.\n")

            traj_rewards = []
            traj_metrics = {"x_err": [], "y_err": [], "jitter": [], "jerk": []}

            for ep in range(episodes_per_traj):
                obs, _ = env.reset(options={"trajectory_mode": traj})
                ep_reward = 0.0
                
                drone_xy_hist, target_xy_hist, drone_vel_hist = [], [], []

                if render:
                    for _ in range(10): p.stepSimulation() 
                
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
            logging.info("") 

    except KeyboardInterrupt:
        logging.info("\nEvaluation manually terminated. Saving collected data...")
    
    finally:
        env.close()
        
    # 4. Save and Print Results
    save_and_print_results(all_results, checkpoint_path)


def save_and_print_results(results: List[Dict], checkpoint_path: str):
    if not results:
        return
        
    ckpt_dir = os.path.dirname(checkpoint_path)
    if not ckpt_dir:
        ckpt_dir = "."
        
    csv_path = os.path.join(ckpt_dir, "evaluation_metrics.csv")
    
    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    logging.info("\n" + "="*85)
    logging.info(f"{'TRAJECTORY':<15} | {'REWARD':<10} | {'X-ERR (m)':<10} | {'Y-ERR (m)':<10} | {'JITTER':<10} | {'JERK RMS':<10}")
    logging.info("-" * 85)
    for r in results:
        logging.info(f"{r['trajectory']:<15} | {r['avg_reward']:<10.1f} | {r['avg_x_err']:<10.3f} | {r['avg_y_err']:<10.3f} | {r['avg_jitter']:<10.3f} | {r['avg_jerk_rms']:<10.2f}")
    logging.info("="*85)
    logging.info(f"Metrics saved to: {csv_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained TD3 Drone Tracker")
    
    # Checkpoint path
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt file (e.g., final.pt or ep_4000.pt)")
    parser.add_argument("--headless", action="store_true", help="Run without PyBullet GUI (much faster)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes per trajectory")
    
    # --- NEW: Command Line Trajectory Selection ---
    parser.add_argument("--trajectory", type=str, default="teleop", 
                        choices=["teleop", "all", "square", "triangular", "sawtooth", "square_wave", "spline_easy", "spline_medium", "spline_hard"],
                        help="Which trajectory to test. Use 'all' for the standard autonomous suite, or 'teleop' for manual control.")
    
    args = parser.parse_args()
    
    # --- Logic to handle the new argument ---
    if args.trajectory == "all":
        TEST_TRAJECTORIES = [
            "square", 
            "triangular", 
            "sawtooth", 
            "square_wave", 
            "spline_easy", 
            "spline_medium", 
            "spline_hard"
        ]
    else:
        TEST_TRAJECTORIES = [args.trajectory]
    
    # Enforce GUI if teleop is selected (prevents headless teleop crashes)
    render_gui = not args.headless
    if "teleop" in TEST_TRAJECTORIES:
        render_gui = True
    
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint, 
        trajectories=TEST_TRAJECTORIES, 
        episodes_per_traj=args.episodes, 
        render=render_gui 
    )