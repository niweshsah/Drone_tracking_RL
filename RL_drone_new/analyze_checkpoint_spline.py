import argparse
import os
import time
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt

from environment import DroneTrackingEnv
from agent import TD3Agent, TD3Config
from utils import compute_tracking_metrics, ensure_dir

# --- Logging Setup ---
ensure_dir("analysis_reports")
ensure_dir("analysis_reports/plots")  # Directory for the generated PNGs
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("analysis_reports/checkpoint_analysis.log", mode='w'),
        logging.StreamHandler()
    ]
)

def evaluate_trajectory(agent, trajectory_mode, num_episodes=2, max_steps=1000):
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

def generate_trajectory_plots(track_name, telemetry_data, save_dir="analysis_reports/plots"):
    """Generates and saves a 3-panel visualization of the drone's performance."""
    # Take the first episode's telemetry for plotting
    ep_data = telemetry_data[0]
    
    d_pos = ep_data["drone_pos"]
    t_pos = ep_data["target_pos"]
    actions = ep_data["actions"]
    time = ep_data["time"]
    
    # Calculate Euclidean error over time
    error = np.linalg.norm(d_pos - t_pos, axis=1)
    
    # Create a 1x3 subplot figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Performance Analysis: {track_name.upper()}", fontsize=16, fontweight='bold', y=1.05)
    
    # Panel 1: 2D Spatial Path
    axs[0].plot(t_pos[:, 0], t_pos[:, 1], 'r--', linewidth=2, label='Target Path')
    axs[0].plot(d_pos[:, 0], d_pos[:, 1], 'b-', linewidth=1.5, alpha=0.8, label='Drone Path')
    axs[0].set_title("2D Tracking Spatial Path")
    axs[0].set_xlabel("X Position (m)")
    axs[0].set_ylabel("Y Position (m)")
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.7)
    axs[0].axis('equal') # Ensures the trajectory doesn't look stretched
    
    # Panel 2: Tracking Error over Time
    axs[1].plot(time, error, 'k-', linewidth=1.5)
    axs[1].fill_between(time, error, alpha=0.2, color='gray')
    axs[1].set_title("Euclidean Tracking Error")
    axs[1].set_xlabel("Time (seconds)")
    axs[1].set_ylabel("Error Distance (m)")
    axs[1].grid(True, linestyle=':', alpha=0.7)
    
    # Panel 3: Control Effort (Actions) over Time
    axs[2].plot(time, actions[:, 0], label='Action X (Res. Vx)', alpha=0.8)
    axs[2].plot(time, actions[:, 1], label='Action Y (Res. Vy)', alpha=0.8)
    axs[2].set_title("Agent Residual Control Effort")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_ylabel("Normalized Output [-1, 1]")
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{track_name}_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f" Saved performance plot to: {save_path}")

def main():
    parser = argparse.ArgumentParser("Detailed VTD3 Checkpoint Analysis")
    # parser.add_argument("--checkpoint", type=str, default="/home/teaching/RL/checkpoints_spline_windy/ep_1575.pt", help="Path to model weights")
    parser.add_argument("--checkpoint", type=str, default="/home/teaching/RL/checkpoints_spline/final.pt", help="Path to model weights")
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

    # Include your newly added splines if you want to test them as well!
    # trajectories = ["triangular", "square", "sawtooth", "square_wave", "spline_easy", "spline_hard"]
    trajectories = ["spline_easy", "spline_hard"]
    all_telemetry = {}

    for track in trajectories:
        logging.info(f"\nEvaluating Trajectory: [{track.upper()}] ...")
        
        try:
            metrics, telemetry = evaluate_trajectory(agent, track, num_episodes=args.episodes_per_track)
            all_telemetry[track] = telemetry
            
            # --- NEW: Generate and save the matplotlib plots ---
            generate_trajectory_plots(track, telemetry)
            
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
                
        except Exception as e:
            logging.error(f" Failed to evaluate {track}: {e}")

    # Save Telemetry for future manual plotting or deep data science analysis
    npz_path = "analysis_reports/telemetry_data.npz"
    np.savez_compressed(npz_path, **all_telemetry)
    
    logging.info("="*70)
    logging.info(f" Analysis complete. Full report saved to analysis_reports/checkpoint_analysis.log")
    logging.info(f" All trajectory PNG plots saved to analysis_reports/plots/")
    logging.info(f" Raw telemetry exported to {npz_path}")
    logging.info("="*70)

if __name__ == "__main__":
    main()