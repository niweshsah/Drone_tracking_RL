import numpy as np
import pybullet as p
import math
import sys
import os

# --- NEW: Add the parent directory to Python's path ---
# 1. Get the absolute path of the directory containing this script (debugging/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the parent directory (RL_drone_new/)
parent_dir = os.path.dirname(current_dir)
# 3. Add it to sys.path so Python knows to look there for modules
sys.path.append(parent_dir)
# ------------------------------------------------------

# Now you can import as if you were in the parent directory
from environment import DroneTrackingEnv
from train_curriculum import get_curriculum_trajectories, get_curriculum_phase

def test_curriculum_math():
    print("\n--- TEST 1: CURRICULUM LOGIC ---")
    total_episodes = 2000
    random_episodes = 50
    
    # Test Random Phase
    assert get_curriculum_trajectories(10, random_episodes, total_episodes) == ["triangular"], "Random phase should only have triangular"
    
    # Test boundaries (Off-by-one errors)
    learning_total = total_episodes - random_episodes # 1950
    phase_1_start = random_episodes + int(0.25 * learning_total) # 537
    
    assert "square" in get_curriculum_trajectories(phase_1_start + 1, random_episodes, total_episodes), "Phase 1 math failed"
    
    # Test Final Phase
    final_trajectories = get_curriculum_trajectories(1999, random_episodes, total_episodes)
    assert len(final_trajectories) == 4, f"Final phase missing trajectories! Got: {final_trajectories}"
    
    print("✅ Curriculum math is correct. No out-of-bounds or off-by-one errors.")


def test_environment_spaces_and_nans():
    print("\n--- TEST 2: ENVIRONMENT BOUNDS & NaN CHECKS ---")
    env = DroneTrackingEnv(cfg=None, trajectory_mode="triangular", GUI_mode=False)
    print("working below test2")
    state, info = env.reset()
    
    assert not np.isnan(state).any(), "NaN found in initial state!"
    assert np.all((state >= -1.0) & (state <= 1.0)), f"Initial state out of bounds [-1, 1]! Got: {state}"
    
    # Stress test with extreme and out-of-bounds actions
    extreme_actions = [
        np.array([1.0, 1.0], dtype=np.float32),    # Max acceleration
        np.array([-1.0, -1.0], dtype=np.float32),  # Max reverse
        np.array([0.0, 0.0], dtype=np.float32),    # Dead drift
        np.array([50.0, -50.0], dtype=np.float32)  # Severe Out-of-bounds (Should be clipped)
    ]
    
    for act in extreme_actions:
        next_state, reward, done, truncated, info = env.step(act)
        
        # 1. NaN and Inf checks
        assert not np.isnan(next_state).any(), f"NaN state produced by action {act}"
        assert not np.isnan(reward), f"NaN reward produced by action {act}"
        assert not math.isinf(reward), f"Infinite reward produced by action {act}"
        
        # 2. State Space Integrity
        # Allow a tiny epsilon (1e-5) for floating point inaccuracies near 1.0
        assert np.all((next_state >= -1.00001) & (next_state <= 1.00001)), f"Observation space bound broken! State: {next_state}"
        
    print("✅ Environment handles extreme actions safely. State bounds and rewards are stable.")
    env.close()


def test_trajectory_hotswapping():
    print("\n--- TEST 3: DYNAMIC TRAJECTORY RESET ---")
    env = DroneTrackingEnv(cfg=None, trajectory_mode="triangular", GUI_mode=False)
    
    trajectories_to_test = ["triangular", "square", "sawtooth", "square_wave"]
    
    for traj in trajectories_to_test:
        state, info = env.reset(options={"trajectory_mode": traj})
        
        # Ensure the info dict returns the EXACT trajectory requested
        reported_traj = info.get("current_trajectory", None)
        assert reported_traj == traj, f"Requested '{traj}' but info dict reported '{reported_traj}'"
        
        # Step once to ensure the new physics path doesn't crash on init
        env.step(np.array([0.1, -0.1], dtype=np.float32))
        
    print("✅ Hot-swapping trajectories via reset(options=...) works flawlessly.")
    env.close()


def test_truncation_limits():
    print("\n--- TEST 4: EPISODE TRUNCATION LIMITS ---")
    env = DroneTrackingEnv(cfg=None, trajectory_mode="square", GUI_mode=False)
    
    max_steps = env.cfg.max_episode_steps
    env.reset()
    
    truncated = False
    for i in range(max_steps + 5):
        _, _, done, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        if truncated:
            step_count = i + 1
            break
            
    assert truncated == True, "Environment never truncated!"
    assert step_count == max_steps, f"Truncated at step {step_count}, expected {max_steps}!"
    
    print(f"✅ Episode correctly truncates exactly at max_episode_steps ({max_steps}).")
    env.close()


if __name__ == "__main__":
    print("Starting Drone RL Debug Suite...")
    try:
        test_curriculum_math()
        test_environment_spaces_and_nans()
        test_trajectory_hotswapping()
        test_truncation_limits()
        print("\n🏆 ALL TESTS PASSED! Your environment is ready for training.")
    except AssertionError as e:
        print(f"\n❌ FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 UNEXPECTED CRASH: {e}")
        sys.exit(1)