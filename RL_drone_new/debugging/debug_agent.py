# test_agent.py
import os
import torch
import numpy as np
import logging
from RL_drone_new.agent import TD3Agent, TD3Config, ReplayBuffer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', 
                    handlers=[logging.FileHandler("test_agent.log", mode='w'), logging.StreamHandler()])

def run_agent_diagnostics():
    logging.info("=== STARTING AGENT & BUFFER DIAGNOSTICS ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Testing on Device: {device}")

    # 1. Test Replay Buffer Shapes & Sampling
    logging.info("Testing Replay Buffer...")
    buffer = ReplayBuffer(state_dim=6, action_dim=2, max_size=1000)
    
    # Add 100 dummy transitions
    for i in range(100):
        s = np.random.randn(6).astype(np.float32)
        a = np.random.uniform(-1, 1, 2).astype(np.float32)
        r = float(np.random.randn())
        ns = np.random.randn(6).astype(np.float32)
        done = float(i % 10 == 0) # 1.0 every 10 steps
        buffer.add(s, a, r, ns, done)

    assert buffer.size == 100, "Buffer size tracking failed"
    s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size=32, device=torch.device(device))
    
    assert s_b.shape == (32, 6), f"State batch shape wrong: {s_b.shape}"
    assert a_b.shape == (32, 2), f"Action batch shape wrong: {a_b.shape}"
    assert r_b.shape == (32, 1), f"Reward batch shape wrong: {r_b.shape}"
    logging.info("-> Replay Buffer OK.")

    # 2. Test TD3 Agent Initialization and Action Selection
    logging.info("Testing TD3 Agent Forward Pass...")
    cfg = TD3Config(state_dim=6, action_dim=2, max_action=1.0, hidden_dim=256, device=device)
    agent = TD3Agent(cfg)
    
    test_state = np.zeros(6, dtype=np.float32)
    action = agent.select_action(test_state)
    assert action.shape == (2,), f"Action shape wrong: {action.shape}"
    assert -1.0 <= action[0] <= 1.0, "Action out of bounds"
    logging.info(f"-> Agent Action OK. Action: {action}")

    # 3. Test Training Step (Backpropagation & Loss)
    logging.info("Testing TD3 Backpropagation (Train Step)...")
    loss_dict = agent.train_step(buffer, batch_size=32)
    assert "critic_loss" in loss_dict, "Missing critic loss"
    assert "actor_loss" in loss_dict, "Missing actor loss"
    logging.info(f"-> Training Step OK. Losses: {loss_dict}")

    # 4. Test Save/Load Checkpoints
    logging.info("Testing Weight Save/Load...")
    save_path = "test_model_weights.pt"
    agent.save(save_path)
    assert os.path.exists(save_path), "Model failed to save"
    
    # Modify a weight, then load to ensure it restores
    agent.actor.l1.weight.data.fill_(0.0)
    agent.load(save_path)
    assert torch.sum(agent.actor.l1.weight.data) != 0.0, "Model failed to load weights properly"
    
    os.remove(save_path) # Cleanup
    logging.info("-> Model Save/Load OK.")

    logging.info("=== AGENT DIAGNOSTICS PASSED ===")

if __name__ == "__main__":
    run_agent_diagnostics()