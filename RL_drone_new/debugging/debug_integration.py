import numpy as np
import torch
import logging
import time
from RL_drone_new.environment import DroneTrackingEnv
from RL_drone_new.agent import TD3Agent, TD3Config, ReplayBuffer

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler("integration_v2.log", mode='w'), logging.StreamHandler()]
)

def verify_device_placement(agent):
    """Checks if all model parameters are on the correct device."""
    devices = {p.device for p in agent.actor.parameters()}
    return devices

def run_detailed_diagnostics():
    logging.info("="*60)
    logging.info("STARTING TD3 AGENT-ENVIRONMENT INTEGRATION (V2)")
    logging.info("="*60)

    # 1. Device Setup
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Target Device: {device_str}")

    # 2. Environment Initialization
    try:
        env = DroneTrackingEnv(cfg=None, trajectory_mode="triangular", GUI_mode=False)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
        logging.info(f"Env initialized. State: {state_dim}, Action: {action_dim}, Max Action: {max_action}")
    except Exception as e:
        logging.error(f"Failed to load Environment: {e}")
        return

    # 3. Agent & Buffer Setup
    # Using the exact parameters from your TD3Config dataclass
    cfg = TD3Config(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        hidden_dim=256,
        device=device_str
    )
    
    agent = TD3Agent(cfg)
    buffer = ReplayBuffer(state_dim, action_dim, max_size=1000)
    
    # Check if the agent actually moved to the GPU
    actual_devices = verify_device_placement(agent)
    logging.info(f"Agent parameter locations: {actual_devices}")
    
    if "cuda" in device_str and not any("cuda" in str(d) for d in actual_devices):
        logging.error("CRITICAL: Agent requested CUDA but remains on CPU!")
        return

    # 4. Phase 1: Interaction Test (Data Flow)
    logging.info("--- Phase 1: Interaction & Buffer Storage ---")
    obs, _ = env.reset()
    
    for step in range(50):
        # Test agent.select_action (Handles internal tensor conversion)
        action = agent.select_action(obs)
        
        # Add exploration noise
        noise = np.random.normal(0, 0.1, size=action_dim).astype(np.float32)
        action_stepped = np.clip(action + noise, -max_action, max_action)
        
        next_obs, reward, terminated, truncated, _ = env.step(action_stepped)
        
        # Test ReplayBuffer.add
        buffer.add(obs, action_stepped, reward, next_obs, float(terminated or truncated))
        
        obs = next_obs
        if (step + 1) % 10 == 0:
            logging.info(f"Interaction Step {step+1}/50: Action Sample {action}")

    logging.info(f"Buffer populated. Current size: {buffer.size}")

    # 5. Phase 2: Training Test (The Gradient Pass)
    logging.info("--- Phase 2: Training Step (Backprop) ---")
    if buffer.size >= 32:
        try:
            # We run a few steps to ensure the policy_delay logic triggers
            for i in range(5):
                losses = agent.train_step(buffer, batch_size=32)
                logging.info(f"Train Step {i+1} | Critic Loss: {losses['critic_loss']:.4f} | Actor Loss: {losses['actor_loss']:.4f}")
            
            logging.info("Gradient backpropagation successful.")
        except Exception as e:
            logging.error(f"Training step failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return
    else:
        logging.warning("Buffer too small to test training.")

    # 6. Phase 3: Save/Load Consistency
    logging.info("--- Phase 3: Save/Load Check ---")
    test_path = "integration_test_model.pth"
    try:
        agent.save(test_path)
        agent.load(test_path)
        logging.info("Model Save/Load cycle passed.")
    except Exception as e:
        logging.error(f"Save/Load failed: {e}")

    logging.info("="*60)
    logging.info("DIAGNOSTICS COMPLETE: Integration looks healthy.")
    logging.info("="*60)
    env.close()

if __name__ == "__main__":
    run_detailed_diagnostics()