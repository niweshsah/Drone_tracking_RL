import argparse
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import TD3Agent, TD3Config
from environment import DroneTrackingEnv, EnvConfig
from utils import compute_tracking_metrics, ensure_dir, moving_average, write_csv_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate trained VTD3 policy")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--trajectory", type=str, default="square_wave", choices=["triangular", "square", "sawtooth", "square_wave"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="eval_outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_agent(device: str, max_action: float = 6.0) -> TD3Agent:
    cfg = TD3Config(
        state_dim=2,
        action_dim=2,
        max_action=max_action,
        hidden_dim=64,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        device=device,
    )
    return TD3Agent(cfg)


def run_episode(env: DroneTrackingEnv, agent: TD3Agent, max_steps: int) -> Dict[str, np.ndarray]:
    state, info = env.reset()

    drone_pos = [info["drone_pos"][:2].copy()]
    target_pos = [info["target_pos"][:2].copy()]
    drone_vel = [info["drone_vel"][:2].copy()]
    rewards = []
    state_err = [float(np.linalg.norm(state))]

    done = False
    truncated = False
    steps = 0

    while not (done or truncated):
        action = agent.select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state

        drone_pos.append(info["drone_pos"][:2].copy())
        target_pos.append(info["target_pos"][:2].copy())
        drone_vel.append(info["drone_vel"][:2].copy())
        rewards.append(reward)
        state_err.append(float(np.linalg.norm(state)))

        steps += 1
        if steps >= max_steps:
            truncated = True

    return {
        "drone_pos": np.asarray(drone_pos, dtype=np.float32),
        "target_pos": np.asarray(target_pos, dtype=np.float32),
        "drone_vel": np.asarray(drone_vel, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "state_err": np.asarray(state_err, dtype=np.float32),
    }


def plot_trajectories(drone_xy: np.ndarray, target_xy: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.plot(target_xy[:, 0], target_xy[:, 1], "r-", linewidth=2, label="Target")
    plt.plot(drone_xy[:, 0], drone_xy[:, 1], "b-", linewidth=2, label="Drone")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rewards_curve(reward_series: np.ndarray, out_path: str) -> None:
    ma = moving_average(reward_series, window=20)
    plt.figure(figsize=(7, 4))
    plt.plot(reward_series, label="Episode Reward", alpha=0.6)
    plt.plot(ma, label="Moving Average", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Reward Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    env_cfg = EnvConfig(seed=args.seed, max_episode_steps=args.max_steps)
    env = DroneTrackingEnv(env_cfg, trajectory_mode=args.trajectory)

    agent = build_agent(args.device)
    agent.load(args.checkpoint)

    metrics_csv = os.path.join(args.out_dir, "metrics.csv")
    traj_npz = os.path.join(args.out_dir, "trajectories.npz")

    all_rewards = []
    all_drone = []
    all_target = []
    all_vel = []

    for ep in range(args.episodes):
        env.seed(args.seed + ep)
        data = run_episode(env, agent, args.max_steps)
        m = compute_tracking_metrics(data["drone_pos"], data["target_pos"], data["drone_vel"], env_cfg.control_period)

        ep_reward = float(np.sum(data["rewards"]))
        all_rewards.append(ep_reward)
        all_drone.append(data["drone_pos"])
        all_target.append(data["target_pos"])
        all_vel.append(data["drone_vel"])

        row = {
            "episode": ep,
            "reward": ep_reward,
            **m,
        }
        write_csv_row(metrics_csv, row)

        plot_trajectories(
            data["drone_pos"],
            data["target_pos"],
            os.path.join(args.out_dir, f"trajectory_ep{ep:03d}.png"),
            title=f"Trajectory - Episode {ep}",
        )

        print(
            f"eval_ep={ep:03d} reward={ep_reward:9.3f} "
            f"x_err={m['x_tracking_error']:.3f} y_err={m['y_tracking_error']:.3f} "
            f"jitter={m['velocity_jitter']:.4f} jerk_rms={m['jerk_rms']:.4f}"
        )

    np.savez(
        traj_npz,
        drone=np.array(all_drone, dtype=object),
        target=np.array(all_target, dtype=object),
        vel=np.array(all_vel, dtype=object),
        rewards=np.array(all_rewards, dtype=np.float32),
    )

    plot_rewards_curve(np.asarray(all_rewards, dtype=np.float32), os.path.join(args.out_dir, "rewards_eval.png"))

    env.close()
    print(f"Evaluation complete. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
