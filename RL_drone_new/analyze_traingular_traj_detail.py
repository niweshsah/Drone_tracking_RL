import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from environment import DroneTrackingEnv
from agent import TD3Agent, TD3Config
from utils import ensure_dir

def run_forensic_analysis(checkpoint_path="/home/teaching/RL/checkpoints_updated_reward_updated_speed/best.pt"):
    print("="*60)
    print(" 🚁 FLIGHT RECORDER: TRIANGULAR TRAJECTORY FORENSICS")
    print("="*60)

    # 1. Initialize Headless Environment & Agent
    env = DroneTrackingEnv(cfg=None, trajectory_mode="triangular", GUI_mode=False)
    
    td3_cfg = TD3Config(state_dim=6, action_dim=2, max_action=1.0, hidden_dim=256, device="cpu")
    agent = TD3Agent(td3_cfg)
    
    try:
        agent.load(checkpoint_path, strict=False)
        agent.actor.eval()
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # 2. Telemetry Buffers
    telemetry = {
        "time": [], "dist_err": [], 
        "target_x": [], "target_y": [], 
        "drone_x": [], "drone_y": [],
        "target_speed": [], "drone_speed": [],
        "action_x": [], "action_y": []
    }

    obs, _ = env.reset(seed=42) # Fixed seed for reproducibility
    
    target_lost_flag = False
    warning_zone_flag = False
    
    print(f"{'Time(s)':<10} | {'Dist Err':<10} | {'Drone Spd':<10} | {'Target Spd':<10} | {'Action [X, Y]'}")
    print("-" * 65)

    # 3. Step-by-Step Execution
    for step in range(1000): # 500 steps * 0.05s = 25 seconds (covers the first few corners)
        t = step * env.cfg.control_period
        
        action = agent.select_action(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        
        # Calculate true physical distance
        dist = np.linalg.norm(env.drone_pos[:2] - env.target_pos[:2])
        drone_spd = np.linalg.norm(env.drone_vel[:2])
        target_spd = np.linalg.norm(env.target_vel[:2])
        
        # Save telemetry
        telemetry["time"].append(t)
        telemetry["dist_err"].append(dist)
        telemetry["target_x"].append(env.target_pos[0])
        telemetry["target_y"].append(env.target_pos[1])
        telemetry["drone_x"].append(env.drone_pos[0])
        telemetry["drone_y"].append(env.drone_pos[1])
        telemetry["drone_speed"].append(drone_spd)
        telemetry["target_speed"].append(target_spd)
        telemetry["action_x"].append(action[0])
        telemetry["action_y"].append(action[1])

        # --- LIVE FORENSIC ALERTS ---
        if dist > 7.0 and not warning_zone_flag:
            print(f"⚠️ {t:05.2f}s  | {dist:05.2f}m   | {drone_spd:05.2f}m/s    | {target_spd:05.2f}m/s    | ENTERED WARNING ZONE (>7m)")
            warning_zone_flag = True
            
        if dist >= 9.8 and not target_lost_flag:
            print(f"🚨 {t:05.2f}s  | {dist:05.2f}m   | {drone_spd:05.2f}m/s    | {target_spd:05.2f}m/s    | TARGET LOST (FOV EXCEEDED)!")
            target_lost_flag = True
            
        # Print every 1 second of simulation time
        if step % 20 == 0:
            print(f"{t:05.2f}s     | {dist:05.2f}m     | {drone_spd:05.2f}m/s     | {target_spd:05.2f}m/s     | [{action[0]:+0.2f}, {action[1]:+0.2f}]")

        obs = next_obs
        if done or truncated:
            break

    env.close()
    
    # 4. Generate Forensic Plots
    plot_forensics(telemetry)


def plot_forensics(data):
    ensure_dir("analysis_reports")
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle("Flight Recorder: Triangular Trajectory Breakdown", fontsize=16, fontweight='bold')

    # Plot 1: 2D Overhead Map
    axs[0].plot(data["target_x"], data["target_y"], 'r--', label='Target Path', linewidth=2)
    axs[0].plot(data["drone_x"], data["drone_y"], 'b-', label='Drone Path', linewidth=2)
    axs[0].set_title("Overhead Trajectory (X, Y)")
    axs[0].set_aspect('equal')
    axs[0].legend()
    axs[0].grid(True, linestyle=':', alpha=0.6)

    # Plot 2: Distance Error & Speeds
    axs[1].plot(data["time"], data["dist_err"], 'k-', label='Distance Error (m)', linewidth=2)
    axs[1].axhline(y=7.0, color='orange', linestyle='--', label='Warning Zone (7m)')
    axs[1].axhline(y=10.0, color='r', linestyle='--', label='FOV Lost (10m)')
    axs[1].set_title("Distance Error Over Time")
    axs[1].set_ylabel("Meters")
    axs[1].legend()
    axs[1].grid(True, linestyle=':', alpha=0.6)

    # Plot 3: RL Residual Actions
    axs[2].plot(data["time"], data["action_x"], 'g-', label='Action X (Residual)', alpha=0.8)
    axs[2].plot(data["time"], data["action_y"], 'm-', label='Action Y (Residual)', alpha=0.8)
    axs[2].set_title("Neural Network Output (Residual Control Effort)")
    axs[2].set_xlabel("Time (Seconds)")
    axs[2].set_ylabel("Action Intensity [-1, 1]")
    axs[2].legend()
    axs[2].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    save_path = "analysis_reports/triangular_forensics.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n📊 Forensic plots saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_forensic_analysis()