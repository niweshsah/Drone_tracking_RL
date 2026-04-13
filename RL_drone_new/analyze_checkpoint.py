import argparse
import os
import time
import logging
import numpy as np
import torch

from environment import DroneTrackingEnv
from agent import TD3Agent, TD3Config
from utils import compute_tracking_metrics, ensure_dir

# --- Logging Setup ---
ensure_dir("analysis_reports")
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("analysis_reports/checkpoint_analysis.log", mode='w'),
        logging.StreamHandler()
    ]
)

def evaluate_trajectory(agent, trajectory_mode, num_episodes=5, max_steps=1000):
    """Runs a strict, noiseless evaluation on a specific trajectory."""
    
    # Initialize Headless Environment for high-speed analysis
    env = DroneTrackingEnv(cfg=None, trajectory_mode=trajectory_mode, GUI_mode=False)
    dt = env.cfg.control_period
    
    ep_metrics = []
    raw_telemetry = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=42 + ep)
        
        # Telemetry buffers for this episode
        drone_pos, target_pos, drone_vel, actions = [], [], [], []
        
        done, truncated = False, False
        step = 0
        
        while not (done or truncated) and step < max_steps:
            # Pure policy: No noise, fully deterministic
            action = agent.select_action(obs)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Record Telemetry
            drone_pos.append(env.drone_pos[:2].copy())
            target_pos.append(env.target_pos[:2].copy())
            drone_vel.append(env.drone_vel[:2].copy())
            actions.append(action.copy())
            
            obs = next_obs
            step += 1
            
        # Convert to numpy arrays for math
        d_pos_arr = np.array(drone_pos)
        t_pos_arr = np.array(target_pos)
        d_vel_arr = np.array(drone_vel)
        act_arr = np.array(actions)
        
        # Calculate Episode Metrics
        errors = d_pos_arr - t_pos_arr
        euclidean_errors = np.linalg.norm(errors, axis=1)
        
        mae = np.mean(euclidean_errors)
        rmse = np.sqrt(np.mean(euclidean_errors**2))
        peak_err = np.max(euclidean_errors)
        control_effort = np.mean(np.linalg.norm(act_arr, axis=1)) # L2 norm of actions
        
        kinematics = compute_tracking_metrics(d_pos_arr, t_pos_arr, d_vel_arr, dt)
        
        ep_metrics.append({
            "mae": mae, "rmse": rmse, "peak_err": peak_err,
            "control_effort": control_effort,
            "jitter": kinematics["velocity_jitter"],
            "jerk_rms": kinematics["jerk_rms"]
        })
        
        raw_telemetry.append({
            "drone_pos": d_pos_arr, "target_pos": t_pos_arr, 
            "drone_vel": d_vel_arr, "actions": act_arr, "time": np.arange(step)*dt
        })
        
    env.close()
    
    # Aggregate Metrics across all episodes
    agg_metrics = {k: np.mean([ep[k] for ep in ep_metrics]) for k in ep_metrics[0].keys()}
    return agg_metrics, raw_telemetry


def main():
    parser = argparse.ArgumentParser("Detailed VTD3 Checkpoint Analysis")
    parser.add_argument("--checkpoint", type=str, default="/home/teaching/RL/checkpoints_new/ep_1800.pt", help="Path to model weights")
    parser.add_argument("--episodes-per-track", type=int, default=5, help="Episodes to run per trajectory")
    args = parser.parse_args()

    logging.info("="*70)
    logging.info(f" CHECKPOINT ANALYSIS REPORT: {args.checkpoint}")
    logging.info("="*70)

    if not os.path.exists(args.checkpoint):
        logging.error(f"Checkpoint not found at {args.checkpoint}")
        return

    # Initialize Agent Architecture (Matches train.py)
    td3_cfg = TD3Config(state_dim=6, action_dim=2, max_action=1.0, hidden_dim=256, device="cpu")
    agent = TD3Agent(td3_cfg)
    agent.load(args.checkpoint, strict=False)
    agent.actor.eval()

    # The Trajectories to test
    trajectories = ["triangular", "square", "sawtooth", "square_wave"]
    all_telemetry = {}

    for track in trajectories:
        logging.info(f"\nEvaluating Trajectory: [{track.upper()}] ...")
        
        metrics, telemetry = evaluate_trajectory(agent, track, num_episodes=args.episodes_per_track)
        all_telemetry[track] = telemetry
        
        logging.info("-" * 50)
        logging.info(f" Mean Absolute Error (MAE) : {metrics['mae']:.4f} meters")
        logging.info(f" Root Mean Sq Error (RMSE) : {metrics['rmse']:.4f} meters")
        logging.info(f" Peak Tracking Error       : {metrics['peak_err']:.4f} meters")
        logging.info(f" Avg Residual Control L2   : {metrics['control_effort']:.4f} (0.0=Perfect Feedforward)")
        logging.info(f" Velocity Jitter           : {metrics['jitter']:.4f} m/s")
        logging.info(f" Jerk (RMS)                : {metrics['jerk_rms']:.4f} m/s^3")
        logging.info("-" * 50)

        # Basic Sanity Evaluation
        if metrics['mae'] < 0.5:
            logging.info(" Verdict: EXCELLENT tracking accuracy.")
        elif metrics['mae'] < 1.0:
            logging.info(" Verdict: ACCEPTABLE tracking, but some drift present.")
        else:
            logging.info(" Verdict: POOR tracking. Policy struggles with this trajectory.")

    # Save Telemetry for future plotting
    npz_path = "analysis_reports/telemetry_data.npz"
    np.savez_compressed(npz_path, **all_telemetry)
    
    logging.info("="*70)
    logging.info(f" Analysis complete. Full report saved to analysis_reports/checkpoint_analysis.log")
    logging.info(f" Raw telemetry exported to {npz_path} for plotting.")
    logging.info("="*70)

if __name__ == "__main__":
    main()