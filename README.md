# VTD3 (Vision-Based TD3) in Headless PyBullet

This repository implements a headless, server-friendly approximation of the VTD3 framework from the paper:

> Zhao et al., "A Vision-Based End-to-End Reinforcement Learning Framework for Drone Target Tracking", Drones 2024.

## What is implemented

- Headless PyBullet simulation using `p.connect(p.DIRECT)`.
- Simplified quadcopter and moving target with four trajectories:
  - triangular
  - square
  - sawtooth
  - square wave (with three occlusion walls)
- Vision-only state representation:
  - $s_1=(x_{box}-x_{des})/x_{des}$
  - $s_2=(S_{box}-S_{des})/S_{des}$
- Analytic bbox generation (YOLOv8 + BoT-SORT approximation) without GUI rendering.
- TD3 controller with:
  - actor (2 hidden layers, 64 units each, ReLU)
  - twin critics
  - target policy smoothing
  - clipped double Q-learning
  - delayed policy update
  - replay buffer
- Three-stage training schedule:
  - random exploration
  - noisy TD3 exploration
  - pure policy TD3
- Offline evaluation and plotting (no on-screen rendering).

## File structure

- `environment.py`: headless PyBullet environment, drone/target dynamics, reward implementation.
- `vision.py`: analytic bbox and tracking-noise simulation.
- `agent.py`: TD3 networks, replay buffer, update logic.
- `train.py`: server-oriented training loop with CSV/TensorBoard/checkpoints.
- `evaluate.py`: evaluation, trajectory export, metrics and plot generation.
- `utils.py`: config, metrics, logging helpers.

## Install

```bash
pip install pybullet gymnasium torch numpy matplotlib
```

## Train

```bash
python train.py
```

Optional example:

```bash
python train.py --trajectory square_wave --episodes 500 --max-steps 400 --device cuda
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/best.pt --trajectory square_wave
```

## Headless/server notes

- No GUI rendering is required at train/eval time.
- Works over SSH on Linux servers.
- CUDA is used when available (`--device cuda`), otherwise CPU.
- Deterministic behavior is supported via fixed seeds.

## Paper-faithful assumptions and simplifications

- YOLOv8 + BoT-SORT is replaced with analytic bbox simulation from relative geometry, with optional Gaussian noise/dropout.
- Distance is proxied by bbox area instead of true depth sensing.
- No ROS/PX4/MAVLink integration in this codebase.
- Drone dynamics are simplified to stable velocity-control kinematics in PyBullet DIRECT mode.
- Reward follows the paper structure:
  - $R=W_1R_s + W_2R_{speed} + W_3R_{stability}$
  - where $R_s$ penalizes state-error growth and rewards reduction
  - $R_{speed}$ encourages fast movement when far
  - $R_{stability}$ encourages low speed near target
- The paper specifies reward structure clearly but does not provide fixed numeric values for `W1, W2, W3`; this implementation uses configurable defaults.

## Outputs generated

Training creates:
- `runs/vtd3/train_log.csv`
- TensorBoard logs (if available)
- `checkpoints/best.pt`, periodic checkpoints, and `checkpoints/final.pt`

Evaluation creates:
- `eval_outputs/metrics.csv`
- `eval_outputs/trajectories.npz`
- trajectory plots (`trajectory_epXXX.png`)
- reward summary plot (`rewards_eval.png`)
