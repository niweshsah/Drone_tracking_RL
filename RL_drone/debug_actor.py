import torch
import numpy as np
from actor_fwd import Actor # Ensure this matches your file name

def test_actor_logic(checkpoint_path):
    # 1. Setup
    max_action = 6.0
    actor = Actor(state_dim=2, action_dim=2, max_action=max_action)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint["actor"] if "actor" in checkpoint else checkpoint
        actor.load_state_dict(state_dict)
        actor.eval()
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # 2. Define Test Cases (s1: x_error, s2: area_error)
    # Note: s1 < 0 (Left), s1 > 0 (Right)
    #       s2 < 0 (Far),  s2 > 0 (Too Close)
    test_scenarios = {
        "Far & Centered":    [ 0.0, -0.8], # s1=0 (center), s2=-0.8 (too far)
        "Far & Left":        [-0.5, -0.8], # Target is left and far
        "Far & Right":       [ 0.5, -0.8], # Target is right and far
        "Close & Centered":  [ 0.0,  0.5], # Target is too close (overshot)
        "Perfect Center":    [ 0.0,  0.0], # Target is exactly where we want it
    }

    print(f"\n{'SCENARIO':<20} | {'INPUT (s1, s2)':<15} | {'OUTPUT (fb, lr)':<18} | {'INTERPRETATION'}")
    print("-" * 85)

    for name, s_input in test_scenarios.items():
        state_t = torch.FloatTensor(s_input).unsqueeze(0)
        with torch.no_grad():
            action = actor(state_t).numpy()[0]
        
        fb, lr = action[0], action[1]
        
        # Determine Logic
        fb_desc = "FORWARD" if fb > 0.5 else ("BACKWARD" if fb < -0.5 else "HOVER")
        lr_desc = "STEER RIGHT" if lr > 0.5 else ("STEER LEFT" if lr < -0.5 else "STRAIGHT")
        
        print(f"{name:<20} | ({s_input[0]:>4.1f}, {s_input[1]:>4.1f}) | ({fb:>5.2f}, {lr:>5.2f}) | {fb_desc}, {lr_desc}")

if __name__ == "__main__":
    # Test your best checkpoint
    test_actor_logic("/home/teaching/RL/checkpoints/best.pt")