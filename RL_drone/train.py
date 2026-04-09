import argparse
import os
from typing import Dict

import numpy as np
import torch
import pybullet as p

# Core VTD3 components
from agent import ReplayBuffer, TD3Agent, TD3Config
from environment import DroneTrackingEnv, EnvConfig
from utils import TrainConfig, ensure_dir, linear_schedule, set_global_seeds, write_csv_row
from recorder import DroneRecorder  # Custom utility for training visualization

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VTD3 in headless PyBullet")
    parser.add_argument("--trajectory", type=str, default="triangular", 
                        choices=["triangular", "square", "sawtooth", "square_wave"])
    parser.add_argument("--episodes", type=int, default=2000) # Total episodes from paper
    parser.add_argument("--video-interval", type=int, default=50) # Record every 50 episodes post-exploration
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=str, default="runs/vtd3")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Hyperparameters from Table 2 of the VTD3 paper [cite: 626]
    cfg = TrainConfig(
        seed=args.seed, 
        device=args.device, 
        total_episodes=args.episodes,
        max_episode_steps=300 # Standard episode length for tracking tasks
    )
    cfg.log_dir = args.log_dir
    cfg.checkpoint_dir = args.checkpoint_dir

    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.checkpoint_dir)
    set_global_seeds(cfg.seed)

    # Initialize environment with VTD3 control period (0.3s) [cite: 626]
    env_cfg = EnvConfig(
        seed=cfg.seed,
        control_period=cfg.control_period, 
        max_episode_steps=cfg.max_episode_steps,
        max_action=cfg.max_action,         # 6 m/s [cite: 626]
        x_des=cfg.x_des,                   # Desired horizontal position 
        s_des=cfg.s_des,                   # Desired target area (distance proxy) 
        w1=cfg.w1, w2=cfg.w2, w3=cfg.w3    # Reward weights [cite: 370]
    )
    env = DroneTrackingEnv(env_cfg, trajectory_mode=args.trajectory)

    # TD3 Agent Configuration [cite: 626]
    td3_cfg = TD3Config(
        state_dim=cfg.state_dim, action_dim=cfg.action_dim,
        max_action=cfg.max_action, hidden_dim=cfg.hidden_dim,
        actor_lr=cfg.actor_lr, critic_lr=cfg.critic_lr,
        gamma=cfg.gamma, tau=cfg.tau, device=cfg.device
    )

    agent = TD3Agent(td3_cfg)
    replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)
    recorder = DroneRecorder(base_dir=os.path.join(args.log_dir, "videos"))

    writer = None 
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        pass

    csv_path = os.path.join(cfg.log_dir, "train_log.csv")
    global_step = 0
    best_reward = -np.inf
    
    # Start the Three-Stage Training Strategy 
    for ep in range(cfg.total_episodes): 
        state, _ = env.reset(seed=cfg.seed + ep) # Periodic target alteration [cite: 604]
        ep_reward = 0.0
        ep_track_err = []
        ep_actor_loss, ep_critic_loss = [], []

        # Logic for Video Recording (Only in Stage 3: Pure Policy) 
        is_pure_policy = ep >= (cfg.random_episodes + cfg.noise_episodes)
        should_record = is_pure_policy and (ep % args.video_interval == 0)

        if should_record:
            recorder.start(ep, args.trajectory)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # Headless capture

        done = False
        truncated = False
        while not (done or truncated):
            
            # Action Selection based on the 3-stage schedule [cite: 609, 665]
            if ep < cfg.random_episodes:
                # STAGE 1: Random Exploration (Episodes 0-1000) [cite: 597]
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.select_action(state)
                # STAGE 2: Noisy Exploration (Episodes 1000-1500) [cite: 600]
                if ep < (cfg.random_episodes + cfg.noise_episodes):
                    noise_std = linear_schedule(cfg.explore_noise, cfg.explore_noise_decay, ep - cfg.random_episodes)
                    action += np.random.normal(0.0, noise_std, size=cfg.action_dim).astype(np.float32)
                
                # STAGE 3: Pure Policy (Episodes 1500-2000) 
                action = np.clip(action, -cfg.max_action, cfg.max_action)

            # Interact with Environment
            next_state, reward, done, truncated, info = env.step(action)
            replay.add(state, action, reward, next_state, float(done or truncated))

            # Visualization for recording
            if should_record:
                p.resetDebugVisualizerCamera(
                    cameraDistance=20.0, cameraYaw=45, cameraPitch=-45,
                    cameraTargetPosition=env.drone_pos.tolist()
                )
                recorder.add_visual_aids(env.drone_pos, env.target_pos, cfg.control_period)

            state = next_state
            ep_reward += reward
            ep_track_err.append(float(np.linalg.norm(state)))

            # Delayed Policy Updates every 50 steps [cite: 626]
            if replay.size >= cfg.batch_size and (global_step % cfg.update_interval == 0):
                for _ in range(cfg.update_interval):
                    loss_dict = agent.train_step(replay, cfg.batch_size)
                    ep_actor_loss.append(loss_dict["actor_loss"])
                    ep_critic_loss.append(loss_dict["critic_loss"])

            global_step += 1

        if should_record:
            recorder.stop()

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

        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(cfg.checkpoint_dir, "best.pt"))

        if (ep + 1) % 50 == 0:
            agent.save(os.path.join(cfg.checkpoint_dir, f"ep_{ep + 1:04d}.pt"))
            print(f"Episode {ep:04d} | Reward: {ep_reward:9.3f} | Error: {avg_track_err:7.4f}")

    agent.save(os.path.join(cfg.checkpoint_dir, "final.pt"))
    if writer is not None: writer.close()
    env.close()

if __name__ == "__main__":
    main()