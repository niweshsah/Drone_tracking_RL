import argparse
import os
from typing import Dict

import numpy as np
import torch

from agent import ReplayBuffer, TD3Agent, TD3Config
from environment import DroneTrackingEnv, EnvConfig
from utils import TrainConfig, ensure_dir, linear_schedule, set_global_seeds, write_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train VTD3 in headless PyBullet")
    parser.add_argument("--trajectory", type=str, default="triangular", choices=["triangular", "square", "sawtooth", "square_wave"])
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=str, default="runs/vtd3")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(seed=args.seed, device=args.device, total_episodes=args.episodes, max_episode_steps=args.max_steps)
    cfg.log_dir = args.log_dir
    cfg.checkpoint_dir = args.checkpoint_dir

    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.checkpoint_dir)
    set_global_seeds(cfg.seed)

    env_cfg = EnvConfig(
        seed=cfg.seed,
        control_period=cfg.control_period,
        max_episode_steps=cfg.max_episode_steps,
        max_action=cfg.max_action,
        x_des=cfg.x_des,
        s_des=cfg.s_des,
        w1=cfg.w1,
        w2=cfg.w2,
        w3=cfg.w3,
    )
    env = DroneTrackingEnv(env_cfg, trajectory_mode=args.trajectory)

    td3_cfg = TD3Config(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        max_action=cfg.max_action,
        hidden_dim=cfg.hidden_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_noise=cfg.policy_noise,
        noise_clip=cfg.noise_clip,
        policy_delay=cfg.policy_delay,
        device=cfg.device,
    )

    agent = TD3Agent(td3_cfg)
    replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        writer = None

    csv_path = os.path.join(cfg.log_dir, "train_log.csv")
    global_step = 0

    best_reward = -np.inf
    for ep in range(cfg.total_episodes):
        state, _ = env.reset(seed=cfg.seed + ep)
        ep_reward = 0.0
        ep_track_err = []
        ep_actor_loss = []
        ep_critic_loss = []

        done = False
        truncated = False
        step = 0

        while not (done or truncated):
            if global_step < cfg.random_steps:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = agent.select_action(state)
                if global_step < (cfg.random_steps + cfg.td3_noise_steps):
                    noise_std = linear_schedule(
                        cfg.explore_noise,
                        cfg.explore_noise_decay,
                        max(0, global_step - cfg.random_steps),
                    )
                    action += np.random.normal(0.0, noise_std, size=cfg.action_dim).astype(np.float32)
                action = np.clip(action, -cfg.max_action, cfg.max_action)

            next_state, reward, done, truncated, info = env.step(action)
            replay.add(state, action, reward, next_state, float(done or truncated))

            state = next_state
            ep_reward += reward
            ep_track_err.append(float(np.linalg.norm(state)))

            if replay.size >= cfg.batch_size and (global_step % cfg.update_interval == 0):
                for _ in range(cfg.update_interval):
                    loss_dict: Dict[str, float] = agent.train_step(replay, cfg.batch_size)
                    ep_actor_loss.append(loss_dict["actor_loss"])
                    ep_critic_loss.append(loss_dict["critic_loss"])

            global_step += 1
            step += 1
            if step >= cfg.max_episode_steps:
                truncated = True

        avg_track_err = float(np.mean(ep_track_err)) if ep_track_err else 0.0
        actor_loss = float(np.mean(ep_actor_loss)) if ep_actor_loss else 0.0
        critic_loss = float(np.mean(ep_critic_loss)) if ep_critic_loss else 0.0

        row = {
            "episode": ep,
            "global_step": global_step,
            "episode_reward": float(ep_reward),
            "avg_state_error": avg_track_err,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "trajectory": args.trajectory,
        }
        write_csv_row(csv_path, row)

        if writer is not None:
            writer.add_scalar("train/episode_reward", ep_reward, ep)
            writer.add_scalar("train/avg_state_error", avg_track_err, ep)
            writer.add_scalar("train/actor_loss", actor_loss, ep)
            writer.add_scalar("train/critic_loss", critic_loss, ep)

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
            agent.save(best_path)

        if (ep + 1) % cfg.checkpoint_interval_episodes == 0:
            ckpt = os.path.join(cfg.checkpoint_dir, f"ep_{ep + 1:04d}.pt")
            agent.save(ckpt)

        print(
            f"episode={ep:04d} reward={ep_reward:9.3f} "
            f"avg_state_err={avg_track_err:7.4f} global_step={global_step}"
        )

    final_ckpt = os.path.join(cfg.checkpoint_dir, "final.pt")
    agent.save(final_ckpt)

    if writer is not None:
        writer.close()
    env.close()

    print(f"Training complete. Final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
