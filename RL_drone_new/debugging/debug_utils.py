# test_utils.py
import os
import numpy as np
import logging
from RL_drone_new.utils import (TrainConfig, compute_tracking_metrics, 
                   exponential_decay_schedule, write_csv_row)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', 
                    handlers=[logging.FileHandler("test_utils.log", mode='w'), logging.StreamHandler()])

def run_utils_diagnostics():
    logging.info("=== STARTING UTILS DIAGNOSTICS ===")
    
    # 1. Test TrainConfig
    logging.info("Testing TrainConfig initialization...")
    cfg = TrainConfig()
    assert cfg.state_dim == 6, f"Expected state_dim 6, got {cfg.state_dim}"
    assert cfg.max_residual_vel == 2.0, "Missing max_residual_vel from config"
    logging.info("-> TrainConfig OK.")

    # 2. Test Exponential Decay Math
    logging.info("Testing Exponential Decay Schedule...")
    val_0 = exponential_decay_schedule(0.15, 0.998, 0)
    val_100 = exponential_decay_schedule(0.15, 0.998, 100)
    assert val_0 == 0.15, "Decay at step 0 should be start value"
    assert val_100 < 0.15, "Decay at step 100 should be less than start value"
    logging.info(f"-> Decay OK. (Step 0: {val_0:.4f}, Step 100: {val_100:.4f})")

    # 3. Test Tracking Metrics (Jerk & Jitter)
    logging.info("Testing Tracking Metrics math...")
    # Mock drone moving smoothly
    drone_pos = np.array([[0,0], [1,1], [2,2], [3,3]])
    target_pos = np.array([[0.1,0.1], [1.1,1.1], [2.1,2.1], [3.1,3.1]])
    drone_vel = np.array([[1,1], [1,1], [1,1], [1,1]]) # Constant velocity
    
    metrics = compute_tracking_metrics(drone_pos, target_pos, drone_vel, dt=0.05)
    assert np.isclose(metrics["x_tracking_error"], 0.1), "X tracking error math failed"
    assert metrics["velocity_jitter"] == 0.0, "Constant velocity should have 0 jitter"
    assert metrics["jerk_rms"] == 0.0, "Constant velocity should have 0 jerk"
    logging.info(f"-> Metrics OK. Output: {metrics}")

    # 4. Test CSV Logger
    logging.info("Testing CSV Write...")
    csv_path = "test_temp_log.csv"
    if os.path.exists(csv_path): os.remove(csv_path)
    
    write_csv_row(csv_path, {"ep": 1, "loss": 0.5})
    write_csv_row(csv_path, {"ep": 2, "loss": 0.4})
    
    assert os.path.exists(csv_path), "CSV file was not created"
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 3, "CSV should have 1 header row + 2 data rows"
    os.remove(csv_path) # Cleanup
    logging.info("-> CSV Logger OK.")

    logging.info("=== UTILS DIAGNOSTICS PASSED ===")

if __name__ == "__main__":
    run_utils_diagnostics()