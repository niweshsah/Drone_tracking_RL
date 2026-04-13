import os
import numpy as np
import matplotlib.pyplot as plt

def plot_alignment_error(npz_path="analysis_reports/telemetry_data.npz", save_path="analysis_reports/alignment_error_plot.png"):
    if not os.path.exists(npz_path):
        print(f"❌ Error: Could not find {npz_path}. Run analyze_checkpoint.py first.")
        return

    print(f"Loading telemetry data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)
    # trajectories = data.files
    trajectories = ["triangular"]

    # Set up a 2x2 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("UAV Target Tracking: Alignment Error Over Time", fontsize=18, fontweight='bold')

    for idx, track in enumerate(trajectories):
        if idx >= 4: 
            break # Just in case there are more than 4, restrict to our 2x2 grid
            
        ax = axes[idx]
        episodes = data[track]
        
        # Plot each episode's error
        for ep_idx, ep_data in enumerate(episodes):
            time_steps = ep_data["time"]
            drone_pos = ep_data["drone_pos"]
            target_pos = ep_data["target_pos"]
            
            # Calculate Euclidean planar error at each timestep: sqrt((x_d - x_t)^2 + (y_d - y_t)^2)
            error = np.linalg.norm(drone_pos - target_pos, axis=1)
            
            # Plot the line (slightly transparent so overlapping episodes are visible)
            ax.plot(time_steps, error, alpha=0.7, label=f'Ep {ep_idx+1}')

        ax.set_title(f"Trajectory: {track.upper()}", fontsize=14)
        ax.set_xlabel("Time (Seconds)", fontsize=12)
        ax.set_ylabel("Positional Error (Meters)", fontsize=12)
        
        # Add a horizontal red dashed line representing the "Acceptable Tracking" threshold (e.g., 1.0 meters)
        ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='1m Threshold')
        
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Make room for the main title
    
    # Save and display
    ensure_dir_exists = os.path.dirname(save_path)
    if ensure_dir_exists and not os.path.exists(ensure_dir_exists):
        os.makedirs(ensure_dir_exists)
        
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_alignment_error()