import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from agent import ReplayBuffer, TD3Agent, TD3Config
from environment import DroneTrackingEnv
from utils import TrainConfig, ensure_dir, exponential_decay_schedule, set_global_seeds, write_csv_row


# ---------------------------------------------------------------------------
# Curriculum definition
# ---------------------------------------------------------------------------

TRAJECTORIES: List[str] = [
    "square", "triangular", "sawtooth", "square_wave", 
    "spline_easy", "spline_medium", "spline_hard"
]


# The ultimate generalization curriculum
CURRICULUM: List[Tuple[float, List[str]]] = [
    
    # 0% - 25%: Master basic physics on predictable paths
    (0.00, ["square", "spline_easy"]),
    
    # 25% - 50%: Introduce sharper geometric turns and moderate randomness
    (0.25, ["square", "triangular", "spline_easy", "spline_medium"]),
    
    # 50% - 75%: Remove easy shapes, force tracking on complex geometries and splines
    (0.50, ["sawtooth", "square_wave", "spline_medium"]),
    
    # 75% - 100%: Pure chaos. Only random, highly aggressive splines.
    (0.75, ["spline_medium", "spline_hard"]),
]



def get_curriculum_trajectories(ep: int, random_episodes: int, total_episodes: int) -> List[str]:
    """
    Returns the set of trajectories available at episode `ep`.

    - During random exploration (ep < random_episodes): always triangular.
      Rationale: we want the replay buffer filled with clean, learnable
      transitions before introducing trajectory complexity.
    - After that: unlock phases as a fraction of the remaining learning window.
    """
    if ep < random_episodes:
        return ["triangular"]

    learning_ep = ep - random_episodes
    learning_total = max(1, total_episodes - random_episodes)
    progress = learning_ep / learning_total  # 0.0 -> 1.0

    available = CURRICULUM[0][1]
    for threshold, trajs in CURRICULUM:
        if progress >= threshold:
            available = trajs
    return available


def get_curriculum_phase(ep: int, random_episodes: int, total_episodes: int) -> int:
    """Returns the current curriculum phase index (0-3) for logging."""
    available = get_curriculum_trajectories(ep, random_episodes, total_episodes)
    return len(available) - 1  # 0=triangular only, 3=all four


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train TD3 drone tracker with curriculum learning")
    # parser.add_argument("--trajectory", type=str, default="curriculum",
    #                     choices=["triangular", "square", "sawtooth", "square_wave", "curriculum"],
    #                     help="'curriculum' enables multi-trajectory curriculum learning; "
    #                          "any single name locks training to that trajectory only.")
    parser.add_argument("--trajectory", type=str, default="curriculum",
                        choices=[
                            "triangular", "square", "sawtooth", "square_wave", 
                            "spline_easy", "spline_medium", "spline_hard", "curriculum"
                        ])
    parser.add_argument("--episodes", type=int, default=4000) # increased for spline curriculum
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-dir", type=str, default="runs/vtd3_curriculum")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_spline")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        seed=args.seed,
        device=args.device,
        total_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
    cfg.log_dir = args.log_dir
    cfg.checkpoint_dir = args.checkpoint_dir

    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.checkpoint_dir)
    set_global_seeds(cfg.seed)

    rng = np.random.default_rng(cfg.seed)

    # Build env once — trajectory is hot-swapped via reset(options=) each episode.
    # We pass "triangular" as a safe default; it will be overridden immediately.
    env = DroneTrackingEnv(cfg=None, trajectory_mode="triangular", GUI_mode=False)

    td3_cfg = TD3Config(
        state_dim=cfg.state_dim,
        action_dim=cfg.action_dim,
        max_action=1.0,
        hidden_dim=cfg.hidden_dim,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        device=cfg.device,
    )

    agent = TD3Agent(td3_cfg)
    replay = ReplayBuffer(cfg.state_dim, cfg.action_dim, cfg.buffer_size)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=cfg.log_dir)
    except Exception:
        print("TensorBoard not available — proceeding without it.")

    csv_path = os.path.join(cfg.log_dir, "train_log.csv")
    global_step = 0
    best_reward = -np.inf

    print(f"=== Training | device: {cfg.device} | mode: {args.trajectory} | episodes: {cfg.total_episodes} ===")

    for ep in range(cfg.total_episodes):

        # ------------------------------------------------------------------
        # Select trajectory for this episode
        # ------------------------------------------------------------------
        if args.trajectory == "curriculum":
            available = get_curriculum_trajectories(ep, cfg.random_episodes, cfg.total_episodes)
            traj = rng.choice(available) 
        else:
            traj = args.trajectory  # single-trajectory mode

        curriculum_phase = get_curriculum_phase(ep, cfg.random_episodes, cfg.total_episodes)

        # Hot-swap the trajectory without rebuilding the PyBullet session.
        state, _ = env.reset(seed=cfg.seed + ep, options={"trajectory_mode": traj})

        ep_reward = 0.0
        ep_track_err = []
        ep_actor_loss, ep_critic_loss = [], []
        done = False
        truncated = False

        while not (done or truncated):

            # --------------------------------------------------------------
            # Three-stage action selection
            # Stage 1 (ep < random_episodes):          pure random
            # Stage 2 (random <= ep < random+noise):   policy + decaying noise
            # Stage 3 (ep >= random+noise):            pure policy
            # --------------------------------------------------------------
            if ep < cfg.random_episodes:
                # Stage 1: fill replay buffer with diverse transitions
                action = env.action_space.sample().astype(np.float32)

            else:
                action = agent.select_action(state)

                if ep < (cfg.random_episodes + cfg.noise_episodes):
                    # Stage 2: noisy policy — noise decays as agent improves
                    noise_std = exponential_decay_schedule(
                        cfg.explore_noise,
                        cfg.explore_noise_decay,
                        ep - cfg.random_episodes,
                    )
                    noise = np.random.normal(0.0, noise_std, size=cfg.action_dim).astype(np.float32)
                    action = np.clip(action + noise, -1.0, 1.0)

                # Stage 3: pure policy — no modification needed

            # Interact with environment
            next_state, reward, done, truncated, info = env.step(action)

            # Only store `done` (not `truncated`) to avoid poisoning the
            # Bellman target at artificial episode boundaries.
            replay.add(state, action, reward, next_state, float(done))

            state = next_state
            ep_reward += reward
            ep_track_err.append(float(np.linalg.norm(state[:2])))

            # Delayed batch updates
            if replay.size >= cfg.batch_size and (global_step % cfg.update_interval == 0):
                for _ in range(cfg.update_interval):
                    loss_dict = agent.train_step(replay, cfg.batch_size)
                    ep_actor_loss.append(loss_dict["actor_loss"])
                    ep_critic_loss.append(loss_dict["critic_loss"])

            global_step += 1

        # ------------------------------------------------------------------
        # Determine exploration stage label for logging
        # ------------------------------------------------------------------
        
                
        if ep < cfg.random_episodes:
            stage_label = "random"
        elif ep < cfg.random_episodes + cfg.noise_episodes:
            stage_label = "noisy"
        else:
            stage_label = "pure"

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        avg_track_err = float(np.mean(ep_track_err)) if ep_track_err else 0.0

        row = {
            "episode": ep,
            "episode_reward": float(ep_reward),
            "avg_state_error": avg_track_err,
            "trajectory": traj,
            "curriculum_phase": curriculum_phase,
            "exploration_stage": stage_label,
        }
        write_csv_row(csv_path, row)

        if writer is not None:
            writer.add_scalar("train/reward", ep_reward, ep)
            writer.add_scalar("train/error", avg_track_err, ep)
            writer.add_scalar("train/curriculum_phase", curriculum_phase, ep)

            # Per-trajectory reward breakdown — shows if any trajectory is lagging
            writer.add_scalar(f"train/reward_{traj}", ep_reward, ep)
            writer.add_scalar(f"train/error_{traj}", avg_track_err, ep)

            if ep_actor_loss:
                writer.add_scalar("train/actor_loss", np.mean(ep_actor_loss), ep)
                writer.add_scalar("train/critic_loss", np.mean(ep_critic_loss), ep)

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(cfg.checkpoint_dir, "best.pt"))

        if (ep + 1) % 25 == 0:
            agent.save(os.path.join(cfg.checkpoint_dir, f"ep_{ep + 1:04d}.pt"))
            print(
                f"Ep {ep + 1:04d}/{cfg.total_episodes} | "
                f"Stage: {stage_label:6s} | "
                f"Phase: {curriculum_phase} ({traj:10s}) | "
                f"Reward: {ep_reward:9.3f} | "
                f"Error: {avg_track_err:7.4f}"
            )

    agent.save(os.path.join(cfg.checkpoint_dir, "final.pt"))
    if writer is not None:
        writer.close()
    env.close()
    print("=== Training complete ===")


if __name__ == "__main__":
    main()