from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

from firebase_helper import FirebaseHelper


class Tools:
    def __init__(self, mini: ReachyMini, firebase: FirebaseHelper):
        self.mini = mini
        self.firebase = firebase

    def move_head(self, direction: str) -> str:
        movement_map = {
            "left": create_head_pose(y=20, mm=True),
            "right": create_head_pose(y=-20, mm=True),
            "center": create_head_pose(y=0, z=0, mm=True),
            "up": create_head_pose(z=20, mm=True),
        }
        if direction in movement_map:
            self.mini.goto_target(head=movement_map[direction], duration=0.5)
        return f"Moved head {direction}"

    def next_example_question(self) -> str:
        self.firebase.log_message("system", "next example question")
        return self.firebase.get_next_example_question()

    def start_quiz(self) -> str:
        self.firebase.log_message("system", "start quiz")
        return "Quiz started"

    def end_conversation(self) -> str:
        self.firebase.log_message("system", "end conversation")
        return "Conversation ended"
