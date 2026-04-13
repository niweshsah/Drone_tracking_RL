"""Centralized configuration for the TD3 UAV tracking project.

This module keeps hyperparameters in dataclasses so all components
(environment wrapper, reward model, and TD3 agent) share the same values.
"""

from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    """Hyperparameters for dense multi-objective reward shaping."""

    c1: float = 1.0
    c2: float = 1.0
    c3: float = 0.1
    c4: float = 1.0
    alpha: float = 3.0
    a_optimal: float = 0.12
    r_safe: float = 0.6
    crash_penalty: float = -20.0


@dataclass
class EnvConfig:
    """Environment and observation construction settings."""

    history_len: int = 4
    frame_dim: int = 6
    action_dim: int = 2
    max_episode_steps: int = 300

    @property
    def state_dim(self) -> int:
        """Final stacked state dimension fed into actor/critic networks."""
        return self.history_len * self.frame_dim


@dataclass
class TD3Config:
    """TD3 algorithm and optimization hyperparameters."""

    gamma: float = 0.99
    tau: float = 0.005

    actor_lr: float = 1e-3
    critic_lr: float = 1e-3

    batch_size: int = 256
    buffer_size: int = 200_000

    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    exploration_noise: float = 0.1


@dataclass
class TrainConfig:
    """Training loop controls and reproducibility settings."""

    seed: int = 7
    episodes: int = 200
    max_steps_per_episode: int = 300

    start_timesteps: int = 1_000
    update_after: int = 1_000
    updates_per_step: int = 1

    checkpoint_every: int = 25
    checkpoint_dir: str = "checkpoints"


@dataclass
class ProjectConfig:
    """Single object that bundles all sub-configurations."""

    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    td3: TD3Config = field(default_factory=TD3Config)
    train: TrainConfig = field(default_factory=TrainConfig)
