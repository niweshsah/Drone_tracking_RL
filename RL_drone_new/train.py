import argparse
import os
from typing import Dict

import numpy as np
import torch
import pybullet as p

# Core VTD3 components
from agent import ReplayBuffer, TD3Agent, TD3Config
from environment import DroneTrackingEnv  # Removed EnvConfig since env handles it internally
from utils import TrainConfig, ensure_dir, exponential_decay_schedule, set_global_seeds, write_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VTD3 in headless PyBullet")
    parser.add_argument("--trajectory", type=str, default="triangular", 
                        choices=["triangular", "square", "sawtooth", "square_wave"])
    parser.add_argument("--episodes", type=int, default=2000) 
    parser.add_argument("--max_episode_steps", type=int, default=1000)
    parser.add_argument("--video-interval", type=int, default=50) 
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=str, default="runs/vtd3")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_updated_reward_updated_speed")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # General Training Hyperparameters
    cfg = TrainConfig(
        seed=args.seed, 
        device=args.device, 
        total_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps 
    )
    
    cfg.log_dir = args.log_dir
    cfg.checkpoint_dir = args.checkpoint_dir

    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.checkpoint_dir)
    set_global_seeds(cfg.seed)

    # Initialize environment
    # We pass cfg=None so the environment uses its built-in default dictionary
    env = DroneTrackingEnv(cfg=None, trajectory_mode=args.trajectory, GUI_mode=False)

    # TD3 Agent Configuration
    td3_cfg = TD3Config(
        state_dim=cfg.state_dim, 
        action_dim=cfg.action_dim,
        max_action=1.0,  # CRITICAL: Forces Actor to output normalized [-1, 1] actions
        hidden_dim=cfg.hidden_dim,
        actor_lr=cfg.actor_lr, 
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma, 
        tau=cfg.tau, 
        device=cfg.device
    )

    agent = TD3Agent(td3_cfg)
    replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)
    


    writer = None 
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        print("TensorBoard not found. Proceeding without TensorBoard logging.")

    csv_path = os.path.join(cfg.log_dir, "train_log.csv")
    global_step = 0
    best_reward = -np.inf
    
    print(f"=== Starting Training on Device: {cfg.device} ===")
    
    # Start the Three-Stage Training Strategy 
    for ep in range(cfg.total_episodes): 
        state, _ = env.reset(seed=cfg.seed + ep) 
        ep_reward = 0.0
        ep_track_err = []
        ep_actor_loss, ep_critic_loss = [], []

        done = False
        truncated = False
        
        while not (done or truncated):
            
            # Action Selection based on schedule
            if ep < cfg.random_episodes:
                # STAGE 1: Pure Random Exploration
                action = env.action_space.sample().astype(np.float32)
            else:
                # Get deterministic action from TD3 Actor
                action = agent.select_action(state)
                
                # STAGE 2: Noisy Exploration
                if ep < (cfg.random_episodes + cfg.noise_episodes):
                    noise_std = exponential_decay_schedule(cfg.explore_noise, cfg.explore_noise_decay, ep - cfg.random_episodes)
                    noise = np.random.normal(0.0, noise_std, size=cfg.action_dim).astype(np.float32)
                    action = np.clip(action + noise, -1.0, 1.0) # Clip to normalized limits
                
                # STAGE 3: Pure Policy relies on strictly bounded outputs (handled by Actor's tanh).

            # Interact with Environment
            next_state, reward, done, truncated, info = env.step(action)
            
            # In your env.py, terminated (done) is hardcoded to False, which is safe.
            # We ONLY pass 'done', never 'truncated', to the replay buffer to avoid Bellman zero-traps.
            replay.add(state, action, reward, next_state, float(done))



            state = next_state
            ep_reward += reward
            ep_track_err.append(float(np.linalg.norm(state[:2]))) # Log error based on planar state features (x, y)

            # Delayed Batch Updates
            if replay.size >= cfg.batch_size and (global_step % cfg.update_interval == 0):
                for _ in range(cfg.update_interval):
                    loss_dict = agent.train_step(replay, cfg.batch_size)
                    ep_actor_loss.append(loss_dict["actor_loss"])
                    ep_critic_loss.append(loss_dict["critic_loss"])

            global_step += 1

        # Logging and Checkpointing
        avg_track_err = float(np.mean(ep_track_err)) if ep_track_err else 0.0
        row = {
            "episode": ep, "episode_reward": float(ep_reward),
            "avg_state_error": avg_track_err, "trajectory": args.trajectory
        }
        write_csv_row(csv_path, row)

        if writer is not None:
            writer.add_scalar("train/reward", ep_reward, ep)
            writer.add_scalar("train/error", avg_track_err, ep)
            if ep_actor_loss:
                writer.add_scalar("train/actor_loss", np.mean(ep_actor_loss), ep)
                writer.add_scalar("train/critic_loss", np.mean(ep_critic_loss), ep)

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(cfg.checkpoint_dir, "best.pt"))

        # Terminal printing every 25 episodes
        if (ep + 1) % 25 == 0:
            agent.save(os.path.join(cfg.checkpoint_dir, f"ep_{ep + 1:04d}.pt"))
            print(f"Episode {ep + 1:04d}/{cfg.total_episodes} | Reward: {ep_reward:9.3f} | Error: {avg_track_err:7.4f}")

    # End of Training
    agent.save(os.path.join(cfg.checkpoint_dir, "final.pt"))
    if writer is not None: 
        writer.close()
    env.close()
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()