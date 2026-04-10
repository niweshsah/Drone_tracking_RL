import numpy as np
import pybullet as p
import time
from environment import DroneTrackingEnv

def visualize_ideal_tracking():
    # 1. Initialize the environment
    env = DroneTrackingEnv(trajectory_mode="triangular")
    env.reset()
    
    # Remove side panels for a cleaner view
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    print("\n--- Visualizing Ideal World: Drone perfectly locked to Target ---")

    try:
        while True:
            # 2. Calculate the target's position at the current time
            target_pos, target_vel = env.trajectory.sample(env.t)
            
            # 3. Create the "Ideal" Drone Position (Directly above the target)
            ideal_drone_pos = np.array([
                target_pos[0], 
                target_pos[1], 
                env.cfg.drone_altitude
            ], dtype=np.float32)

            # 4. OVERRIDE internal state: We force the environment to believe the 
            # drone is already at the ideal spot before the physics step.
            env.drone_pos = ideal_drone_pos.copy()
            env.target_pos = target_pos.copy()

            # 5. Update PyBullet visual bodies
            p.resetBasePositionAndOrientation(env.target_id, target_pos.tolist(), [0,0,0,1])
            p.resetBasePositionAndOrientation(env.drone_id, ideal_drone_pos.tolist(), [0,0,0,1])
            
            # 6. Fixed Follow Camera (Centered on the Drone)
            # Yaw 0/Pitch -89 is almost a perfect top-down "Bird's Eye" view
            p.resetDebugVisualizerCamera(
                cameraDistance=25.0, 
                cameraYaw=45,
                cameraPitch=-40, 
                cameraTargetPosition=ideal_drone_pos.tolist()
            )
            
            # 7. Visual Aids
            # Green Tether
            p.addUserDebugLine(target_pos, ideal_drone_pos, [0, 1, 0], lineWidth=3, lifeTime=env.cfg.control_period)
            # Red breadcrumb path for the target
            p.addUserDebugLine(target_pos, target_pos + [0,0,0.1], [1, 0, 0], lineWidth=5, lifeTime=5.0)

            # 8. Step the environment with a zero-velocity action.
            # We pass zeros because we are manually handling the "Perfect" positioning above.
            dummy_action = np.array([0.0, 0.0], dtype=np.float32)
            env.step(dummy_action)

            time.sleep(env.cfg.control_period)

    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    visualize_ideal_tracking()