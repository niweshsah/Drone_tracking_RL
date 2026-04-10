import os
import torch
import pybullet as p
# import numpy as np
from environment import DroneTrackingEnv, EnvConfig

# import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim=2, action_dim=2, max_action=6.0, hidden_dim=64):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

def record_checkpoint(checkpoint_path, video_name, trajectory="triangular"):
    # Initialize Env in GUI mode for video capture
    env_cfg = EnvConfig(max_action=6.0, control_period=0.3)
    env = DroneTrackingEnv(env_cfg, trajectory_mode=trajectory, render_mode="gui")
    
    # Load Actor
    actor = Actor().to("cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    # Start PyBullet Video Logger
    log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_name)
    
    state, _ = env.reset()
    for _ in range(300): # Standard episode length [cite: 616]
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action = actor(state_t).detach().numpy()[0]
        
        # Birds-eye follow camera [cite: 782]
        p.resetDebugVisualizerCamera(cameraDistance=15.0, cameraYaw=45, 
                                    cameraPitch=-45, cameraTargetPosition=env.drone_pos)
        
        state, reward, done, truncated, _ = env.step(action)
        if done or truncated: break

    p.stopStateLogging(log_id)
    env.close()

if __name__ == "__main__":
    checkpoints = ["checkpoints_dslab/ep_0500.pt", "checkpoints_dslab/ep_1000.pt", 
                   "checkpoints_dslab/ep_1500.pt", "checkpoints_dslab/best.pt"]
    for ckpt in checkpoints:
        if os.path.exists(ckpt):
            name = f"eval_{os.path.basename(ckpt).replace('.pt', '.mp4')}"
            record_checkpoint(ckpt, name)