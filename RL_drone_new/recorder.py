import os
import pybullet as p

class DroneRecorder:
    def __init__(self, base_dir: str = "training_videos"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        self.logging_id = None

    def start(self, episode: int, trajectory: str):
        """Starts MP4 recording for the current episode."""
        path = os.path.join(self.base_dir, f"vtd3_{trajectory}_ep{episode:04d}.mp4")
        # Records at the resolution specified for the paper's dataset (1280x720) [cite: 427]
        self.logging_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path)
        print(f"Recording started: {path}")

    def stop(self):
        """Finalizes the video file."""
        if self.logging_id is not None:
            p.stopStateLogging(self.logging_id)
            self.logging_id = None

    def add_visual_aids(self, drone_pos, target_pos, period):
        """Adds the tether and path lines similar to your ideal reference."""
        # Green Tether between drone and target 
        p.addUserDebugLine(target_pos, drone_pos, [0, 1, 0], lineWidth=3, lifeTime=period)
        # Red breadcrumb path for the target
        p.addUserDebugLine(target_pos, target_pos + [0, 0, 0.1], [1, 0, 0], lineWidth=5, lifeTime=5.0)