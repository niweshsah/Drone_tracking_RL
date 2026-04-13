import numpy as np
from typing import Dict, Tuple






class RewardDrone:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prev_state = None

def _compute_reward(self, s_t: np.ndarray, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        s1_curr, s2_curr = s_t[0], s_t[1]
        
        
        s1_prev, s2_prev = self.prev_state[0], self.prev_state[1]
        fb, lr = action[0], action[1] # fb: forward/backward, lr: left/right

        # 1. Rs: Distance Reward (Existing logic is fine)
        rs = (abs(s1_prev) - abs(s1_curr)) + (abs(s2_prev) - abs(s2_curr))

        # 2. Rspeed: High speed ONLY if moving TOWARD the target [SIGN AWARE]
        rspeed = -10.0
        # If target is too far (s2 < -1.5), we need positive fb (Forward)
        far_cond = (s2_curr < -1.5) and (fb > 0.9 * self.cfg.max_action) # if s2_curr is very negative, we want a strong positive fb action to get closer and fb must be above 90% of max_action to reward high speed towards target
        
        # If target is too close (s2 > 1.5), we need negative fb (Backward)
        close_cond = (s2_curr > 1.5) and (fb < -0.9 * self.cfg.max_action)
        
        if far_cond or close_cond:
            rspeed = 1.0

        # 3. Rstability: Precision near target
        rstability = -10.0
        # If error is small, speed must be low
        if abs(s1_curr) < 0.015 and abs(s2_curr) < 0.02:
            if abs(fb) < 0.1 and abs(lr) < 0.1:
                rstability = 1.0

        total_reward = (self.cfg.w1 * rs) + (self.cfg.w2 * rspeed) + (self.cfg.w3 * rstability)
        return total_reward, {"Rs": rs, "Rspeed": rspeed, "Rstability": rstability}