import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pybullet as p
import pybullet_data # Required for plane.urdf assets
import numpy as np
from environment import DroneTrackingEnv, EnvConfig

# --- Actor Architecture: VTD3 Specifications [cite: 321-325, 621, 626] ---
class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, max_action=6.0, hidden_dim=64):
        super(Actor, self).__init__()
        # Architecture: 2 hidden layers, 64 neurons each [cite: 621, 626]
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        # Activation: ReLU for hidden, tanh for output [cite: 322, 323]
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        # Scaled to max_action (6 m/s) [cite: 324, 626]
        return self.max_action * torch.tanh(self.l3(x))

def view_live(checkpoint_path, trajectory="triangular"):
    # 1. Setup Config (Matches Table 2) [cite: 626]
    cfg = EnvConfig(max_action=6.0, control_period=0.3)
    env = DroneTrackingEnv(cfg, trajectory_mode=trajectory)
    
    # --- GUI RE-CONNECTION PATCH ---
    # Environment defaults to DIRECT; we switch to GUI for your PC
    p.disconnect(env._client)
    env._client = p.connect(p.GUI)
    # Corrected attribute: pybullet_data.getDataPath()
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Clean view
    
    # Re-load visual bodies for the GUI client
    env.plane_id = p.loadURDF("plane.urdf")
    env.drone_id = env._create_drone_body()
    env.target_id = env._create_target_body()
    # -------------------------------

    # 2. Load Policy (Stage 3: Pure Policy) 
    actor = Actor(max_action=cfg.max_action)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Extract state_dict if saved within an agent dictionary
    state_dict = checkpoint["actor"] if "actor" in checkpoint else checkpoint
    actor.load_state_dict(state_dict)
    actor.eval()
    
    print(f"--- Visualizing VTD3 Strategy: {checkpoint_path} ---")
    state, _ = env.reset()
    
    try:
        while True:
            # Action selection without exploration noise [cite: 602]
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_t).numpy()[0]
            
            # Follow Camera: Drone-centric view [cite: 75]
            p.resetDebugVisualizerCamera(
                cameraDistance=12.0, cameraYaw=45, 
                cameraPitch=-35, cameraTargetPosition=env.drone_pos.tolist()
            )
            
            # Execute one control step (0.3s) [cite: 616, 626]
            state, reward, done, truncated, _ = env.step(action)
            
            # Visual marker for tracking tether
            p.addUserDebugLine(env.target_pos, env.drone_pos, [0, 1, 0], lineWidth=2, lifeTime=0.3)
            
            if done or truncated:
                print("Episode end. Resetting environment...")
                state, _ = env.reset()
                
            time.sleep(0.05) # Adjust for real-time visualization
    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":
    # Your best checkpoint represents the fully learned policy [cite: 635, 636]
    view_live("/home/rocinate/Desktop/DL-workspace-pytorch/RL_drone/checkpoints_dslab/best.pt", trajectory="triangular")