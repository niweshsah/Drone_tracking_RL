import numpy as np
# import torch
from vision import VisionBBoxEstimator, VisionConfig, vision_state_from_bbox

def debug_vision_logic():
    cfg = VisionConfig()
    rng = np.random.default_rng(42)
    vision = VisionBBoxEstimator(cfg, rng)
    
    # Test Case 1: Target is 10m directly in front
    drone_pos = np.array([0, 0, 4])
    target_pos = np.array([10, 0, 0]) # Directly ahead
    obs = vision.project_bbox(drone_pos, 0, target_pos)
    print(f"Center Test: Expected ~0.5, Got {obs['x_center']:.2f}")

    # Test Case 2: Target is to the LEFT (+Y)
    target_left = np.array([10, 2, 0])
    obs_left = vision.project_bbox(drone_pos, 0, target_left)
    # Correct VTD3 logic: Left target should be < 0.5
    print(f"Left Test: Expected < 0.5, Got {obs_left['x_center']:.2f}")
    if obs_left['x_center'] > 0.5:
        print("!!! BUG DETECTED: Lateral Inversion. Flip the sign of y_cam.")

    # Test Case 3: Distance Check
    # At 40m, what is the area?
    target_4m = np.array([4, 0, 0])
    obs_4m = vision.project_bbox(drone_pos, 0, target_4m)
    print(f"Area at 4m: {obs_4m['area']:.6f}")
    print(f"Your s_des: {cfg.s_des}")
    if obs_4m['area'] < cfg.s_des:
        print(f"!!! SCALE MISMATCH: Area at 4m is much smaller than s_des.")

if __name__ == "__main__":
    debug_vision_logic()