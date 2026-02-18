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
    while True:
        inp = input("")
        if inp == "exit":
            break
        inp = inp.split(" ")
        if len(inp) < 6:
            inp = [0, 0, 0, 0, 0, 0]
        body_yaw = inp[6] if len(inp) > 6 else 0
        mini.goto_target(
            body_yaw=math.radians(float(body_yaw)),
            head=create_head_pose(x=float(inp[0]), y=float(inp[1]), z=float(inp[2]), roll=float(inp[3]), pitch=float(inp[4]), yaw=float(inp[5]), mm=True, degrees=True),
            duration=2.0
        )