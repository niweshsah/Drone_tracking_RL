import time
import pybullet as p
import logging

from environment import DroneTrackingEnv
from agent import TD3Agent, TD3Config

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def enjoy(checkpoint_path: str, trajectory_mode: str = "triangular"):
    """Loads a trained VTD3 model and visualizes its tracking performance."""
    logging.info("="*50)
    logging.info(f"LOADING TRAINED POLICY: {checkpoint_path}")
    logging.info("="*50)

    # 1. Initialize the Environment in GUI Mode
    # Passing cfg=None forces the environment to use its internal defaults
    env = DroneTrackingEnv(cfg=None, trajectory_mode=trajectory_mode, GUI_mode=True)
    
    # Extract dimensions from the environment
    state_dim = env.observation_space.shape[0]   # Should be 6
    action_dim = env.action_space.shape[0]       # Should be 2
    max_action = 1.0 # Standard normalized action space for TD3

    # 2. Initialize the Agent Architecture (Must match train.py)
    td3_cfg = TD3Config(
        state_dim=state_dim, 
        action_dim=action_dim, 
        max_action=max_action, 
        hidden_dim=256, # Must match the 256 neurons from training!
        device="cpu"    # Inference is fast enough on CPU
    )
    agent = TD3Agent(td3_cfg)

    # 3. Load the Checkpoint
    try:
        # strict=False allows us to load just the actor/critic without breaking 
        # if the optimizer states are slightly different.
        agent.load(checkpoint_path, strict=False)
        agent.actor.eval() # Set network to evaluation mode (disables dropout/batchnorm if any)
        logging.info("✔ Weights loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load checkpoint: {e}")
        env.close()
        return

    # 4. Run the Evaluation Loop
    logging.info("Starting visualization. Press CTRL+C in the terminal to exit.")
    
    # We remove exploration noise entirely for pure policy evaluation
    episodes_to_watch = 5
    
    try:
        for ep in range(episodes_to_watch):
            obs, _ = env.reset()
            ep_reward = 0.0
            step = 0
            
            # Optional: Let the physics engine settle before flying
            for _ in range(10): 
                p.stepSimulation()
            
            done = False
            truncated = False
            
            while not (done or truncated):
                # Request deterministic action from the trained actor
                action = agent.select_action(obs)
                
                # Execute action
                next_obs, reward, done, truncated, _ = env.step(action)
                ep_reward += reward
                obs = next_obs
                step += 1
                
                # Draw a green tether line between the drone and the target
                p.addUserDebugLine(
                    env.target_pos, env.drone_pos, [0, 1, 0], 
                    lineWidth=2.5, lifeTime=env.cfg.control_period
                )
                
                # Slow down the simulation so it looks like real-time
                # (control_period is usually 0.05s)
                time.sleep(env.cfg.control_period)

            logging.info(f"Episode {ep+1} Finished | Steps: {step} | Total Reward: {ep_reward:.2f}")

    except KeyboardInterrupt:
        logging.info("\nVisualization manually terminated by user.")
    
    finally:
        env.close()
        logging.info("Closed PyBullet environment.")

if __name__ == "__main__":
    # Adjust this path if your 'best.pt' is located elsewhere
    # CHECKPOINT = "/home/rocinate/Desktop/DL-workspace-pytorch/RL_drone_new/checkpoints_new/best.pt" 
    CHECKPOINT = "//home/rocinate/Desktop/DL-workspace-pytorch/RL_drone_new/checkpoints_updated_reward/best.pt" 
    
    # You can change this to "square", "sawtooth", or "square_wave" to test generalization!
    TRAJECTORY = "triangular"  
    
    enjoy(CHECKPOINT, TRAJECTORY)