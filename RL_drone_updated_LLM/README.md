# Vision-Based TD3 UAV Target Tracking (PyTorch)

This project provides a clean, modular implementation of a vision-derived
continuous-control RL pipeline for UAV target tracking using TD3.

The implementation focuses on:
- POMDP state construction with temporal stacking (k=4)
- Dense multi-objective reward shaping
- Actor + twin critics in PyTorch
- TD3 core mechanisms (double-Q, target smoothing, delayed policy updates)
- A mock Gymnasium environment that stands in for PyBullet + YOLO integration

## File Overview

- `config.py`: Dataclass-based hyperparameter definitions.
- `reward.py`: Dense reward function components and aggregation.
- `env_wrapper.py`: Observation formatting, temporal stacking, and reward integration.
- `networks.py`: Actor and twin critic network definitions.
- `replay_buffer.py`: Circular replay buffer for off-policy learning.
- `td3_agent.py`: TD3 training logic and target network updates.
- `train.py`: End-to-end training loop with a mock base environment.

## Observation and Action Spaces

- Action: 2D planar velocity command `[v_x, v_y]` bounded to `[-1, 1]`.
- One-frame observation (6D):
  `[x_norm, y_norm, w_norm, h_norm, v_f_norm, v_l_norm]`
- Final state to policy/critics: stacked 4-frame history, shape `(24,)`.

## Reward Design

The wrapper computes reward as:

- Alignment:
  `R_align = c1 * exp(-alpha * sqrt(x_norm^2 + y_norm^2))`
- Scale:
  `R_scale = -c2 * abs((w_norm * h_norm) - A_optimal)`
- Smoothness:
  `R_smooth = -c3 * ||a_t - a_(t-1)||_2^2`
- Boundary:
  linear penalty when radial tracking error exceeds `r_safe`
- Crash:
  large negative scalar when target is lost or out of bounds

All coefficients are configurable via `RewardConfig` in `config.py`.

## Requirements

- Python 3.9+
- torch
- numpy
- gymnasium

Install dependencies:

```bash
pip install torch numpy gymnasium
```

## Run Training

```bash
python train.py
```

This starts a full TD3 loop on a mock environment and periodically writes
checkpoints to `checkpoints/`.

## Integration Notes for Real Simulator

When integrating with gym-pybullet-drones + detector output:
- Keep the wrapped env API unchanged (`reset`, `step`).
- Ensure base env observations can be converted to the 6D normalized frame.
- Populate `info` with `target_lost` and `out_of_bounds` booleans.
- The wrapper and TD3 logic can remain unchanged.
