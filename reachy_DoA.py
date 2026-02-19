from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
from scipy.signal import resample
import time


def doa_to_yaw(doa: float) -> float:
    """Convert ReSpeaker DoA (0=left, pi/2=front, pi=right) to robot yaw (0=front)."""
    return float(doa - (np.pi / 2.0))


with ReachyMini() as mini:
    # Initialization - After this point, both audio devices (input/output) will be seen as busy by other applications!
    mini.goto_target(
        body_yaw=0.0,
        head=create_head_pose(pitch=-0.5, yaw=0.0),
        duration=1.0,
    )
    try:
        while True:
            # Get the latest DoA (Direction of Arrival) estimation from the ReSpeaker microphone array
            doa, is_speech = mini.media.get_DoA()
            if doa is not None:
                yaw = doa_to_yaw(doa)
                print(f"DoA: {doa:.2f} rad, Yaw: {yaw:.2f} rad, Speech: {is_speech}")
                if is_speech:
                    mini.goto_target(
                        body_yaw=yaw,
                        head=create_head_pose(x=0,y=0,z=0,roll=0,pitch=0,yaw=yaw, mm=True, degrees=False),
                        duration=0.5,
                    )
            time.sleep(0.1)  # Adjust the sleep time as needed
    except KeyboardInterrupt:
        print("Exiting...")


    # Release audio devices (input/output)
    mini.media.stop_recording()
    # mini.media.stop_playing()
