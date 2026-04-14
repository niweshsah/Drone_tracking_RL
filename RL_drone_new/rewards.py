import numpy as np
from typing import Dict, Tuple

class RewardDrone:
    def __init__(self):
        self.reward_wts = {
                'align': 2.0, 'scale': -1.5, 'smooth': -0.8, 'energy': -0.1, 'boundary': -1.0, 'crash': -50.0
            }

    # state is [x_n, y_n, w_n, h_n, v_xn, v_yn] where n denotes normalized values
    # xn and yn are in image plane normalized by image width and height, wn and hn are normalized by image dimensions, v_xn and v_yn are normalized velocities
    
      # Reward function now includes:
    # 1. Alignment reward based on distance to target center: r_align = 2.0 * exp(-5.0 * dist_err)
    # 2. Scale reward based on how well the observed area matches desired size: r_scale = -1.5 * abs((w_n * h_n) - s_des)
    # 3. Smoothness penalty on changes in the RESIDUAL action (encouraging smoother corrections): r_smooth = -0.2 * sum((action - prev_action)^2)
    # 4. Optional energy penalty to encourage minimal corrections when well-aligned
    # 5. Boundary penalty for being too far from the target
    # 6. Crash penalty for losing the target
    def _compute_reward(self, state: np.ndarray, action: np.ndarray, target_lost: bool, prev_action: np.ndarray) -> Tuple[float, Dict]:
            
            x_n, y_n, w_n, h_n, _, _ = state # Unpack normalized bbox and velocities
            
            dist_err = np.sqrt(x_n**2 + y_n**2) # Distance error in normalized image space 
            
            wt_dict = self.reward_wts
            w1 , w2, w3, w4, w5, w6 = wt_dict['align'], wt_dict['scale'], wt_dict['smooth'], wt_dict['energy'], wt_dict['boundary'], wt_dict['crash']
            
            r_align = w1 * np.exp(-5.0 * dist_err) # Strong exponential reward for being close to the target center
            
            r_scale = 0 * w2
            # r_scale = -1.5 * abs((w_n * h_n) - self.cfg.s_des) # Penalize deviation from desired scale (size in image)
            # not needed as drone has fix altitude
            
            # The Smoothness penalty now punishes high-frequency changes in the RESIDUAL action
            r_smooth = w3 * np.sum(np.square(action - prev_action)) # Encourage smoother corrections by penalizing large changes in the residual action from one step to the next
            
            # Optional: Add a small energy penalty to incentivize the agent to output [0,0] when perfectly aligned
            r_energy = w4 * np.sum(np.square(action)) # Penalize large corrections to encourage minimal intervention when well-aligned

            r_far =  w5 * max(0, dist_err - 0.7) 
            r_crash = w6 if target_lost else 0.0 # not  considering target loss as a crash in this version, but can be adjusted based on desired behavior

            total = r_align + r_scale + r_smooth + r_energy + r_far + r_crash
            return total, {"align": r_align, "smooth": r_smooth, "crash": r_crash}