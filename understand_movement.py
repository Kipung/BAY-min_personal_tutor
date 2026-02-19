from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import math

with ReachyMini() as mini:
    mini.goto_target(
        body_yaw=0.0,
        head=create_head_pose(z=10, mm=True),
        duration=1.0,
    )
    mini.goto_target(
        body_yaw=0.0,
        head=create_head_pose(z=0, mm=True),
        duration=1.0,
    )
    print(mini.media.get_input_audio_samplerate())
    try:
        while True:
            inp = input("")
            inp = inp.split(" ")
            if len(inp) < 6:
                inp = [0, 0, 0, 0, 0, 0]
            body_yaw = inp[8] if len(inp) > 8 else 0
            antennas = (math.radians(float(inp[6])), math.radians(float(inp[7]))) if len(inp) > 7 else None
            mini.goto_target(
                body_yaw=math.radians(float(body_yaw)),
                head=create_head_pose(x=float(inp[0]), y=float(inp[1]), z=float(inp[2]), roll=float(inp[3]), pitch=float(inp[4]), yaw=float(inp[5]), mm=True, degrees=True),
                duration=0.5,
                antennas=antennas,
            )
    except KeyboardInterrupt:
        pass