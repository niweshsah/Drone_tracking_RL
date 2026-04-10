import torch
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
from environment import DroneTrackingEnv, EnvConfig
from live_view import Actor # Ensure your Actor class is in live_view.py

def run_detailed_analysis(checkpoint_path, trajectory="triangular"):
    # 1. Setup Environment and Policy
    cfg = EnvConfig(max_action=6.0, control_period=0.3)
    env = DroneTrackingEnv(cfg, trajectory_mode=trajectory)
    
    actor = Actor(max_action=cfg.max_action)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["actor"] if "actor" in ckpt else ckpt
    actor.load_state_dict(state_dict)
    actor.eval()

    # Data collection containers
    times = []
    drone_pos, target_pos = [], []
    drone_vel = []
    x_errors, y_errors, z_errors = [], [], []

    print(f"--- Analyzing Checkpoint: {checkpoint_path} ---")
    state, _ = env.reset()
    
    # 2. Run Evaluation Episode
    for i in range(cfg.max_episode_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_t).numpy()[0]
        
        # Capture current positions for analysis
        d_pos = env.drone_pos.copy()
        t_pos = env.target_pos.copy()
        
        # Calculate Errors based on Paper Equations 17, 18, 19
        # X_dif: lateral precision
        x_err = abs(d_pos[0] - t_pos[0]) 
        # Y_dif: longitudinal precision (Target distance: 40m)
        y_err = abs(d_pos[1] - t_pos[1] - 40) 
        # Z_dif: altitude precision (Target altitude: 4m)
        z_err = abs(d_pos[2] - 4.0)

        # Store data
        times.append(i * cfg.control_period)
        drone_pos.append(d_pos)
        target_pos.append(t_pos)
        drone_vel.append(env.drone_vel.copy())
        x_errors.append(x_err)
        y_errors.append(y_err)
        z_errors.append(z_err)

        state, _, done, truncated, _ = env.step(action)
        if done or truncated: break

    # 3. Calculate Performance Metrics
    drone_pos = np.array(drone_pos)
    target_pos = np.array(target_pos)
    drone_vel = np.array(drone_vel)
    
    X_dif = np.mean(x_errors)
    Y_dif = np.mean(y_errors)
    Z_dif = np.mean(z_errors)
    
    # Velocity Jitter (Iv): standard dev of first-order difference
    dv = np.diff(drone_vel, axis=0)
    Iv = np.sqrt(np.mean(dv**2)) 
    
    # Jerk RMS (J_RMS): root mean square of second-order difference
    d2v = np.diff(drone_vel, n=2, axis=0)
    J_RMS = np.sqrt(np.mean((d2v / (cfg.control_period**2))**2))

    # 4. Print Summary Log
    print("\n" + "="*30)
    print(f"VTD3 PERFORMANCE LOG")
    print("-" * 30)
    print(f"X-Axis Avg Error:  {X_dif:.4f} m")
    print(f"Y-Axis Avg Error:  {Y_dif:.4f} m")
    print(f"Altitude Stability: {Z_dif:.4f} m")
    print(f"Velocity Jitter:   {Iv:.4f} m/s")
    print(f"Jerk (Smoothness): {J_RMS:.4f} m/s^3")
    print("="*30 + "\n")

    # 5. Generate Visual Analysis
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"VTD3 Analysis: {trajectory.capitalize()} Trajectory", fontsize=16)

    # Plot 1: XY Trajectory (Top Down)
    axs[0, 0].plot(target_pos[:, 0], target_pos[:, 1], 'r--', label='Target Path')
    axs[0, 0].plot(drone_pos[:, 0], drone_pos[:, 1], 'b-', label='Drone Path')
    axs[0, 0].set_title("XY Plane Trajectory")
    axs[0, 0].legend()

    # Plot 2: Altitude Stability over Time
    axs[0, 1].plot(times, drone_pos[:, 2], 'g-')
    axs[0, 1].axhline(y=4.0, color='r', linestyle='--')
    axs[0, 1].set_title("Altitude (Z) over Time")
    axs[0, 1].set_ylabel("Meters")

    # Plot 3: Tracking Errors over Time
    axs[1, 0].plot(times, x_errors, label='X Error')
    axs[1, 0].plot(times, y_errors, label='Y Error')
    axs[1, 0].set_title("Tracking Deviations")
    axs[1, 0].legend()

    # Plot 4: Velocity Profile
    axs[1, 1].plot(times, drone_vel[:, 0], label='VX')
    axs[1, 1].plot(times, drone_vel[:, 1], label='VY')
    axs[1, 1].set_title("Velocity Profiles (m/s)")
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"analysis_{trajectory}.png")
    plt.show()

    env.close()

if __name__ == "__main__":
    run_detailed_analysis("/home/rocinate/Desktop/DL-workspace-pytorch/RL_drone/checkpoints_dslab/best.pt")