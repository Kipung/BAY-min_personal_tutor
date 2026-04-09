import asyncio
import random

from firebase_helper import FirebaseHelper
from motion import (
    enqueue_head_command, enqueue_pose_command, enqueue_emotion_command,
    EMOTION_CATEGORIES, ALL_EMOTION_NAMES,
)


class Tools:
    def __init__(self, firebase: FirebaseHelper, motion_queue: asyncio.Queue):
        self.firebase = firebase
        self.motion_queue = motion_queue

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------

    def move_head(
        self,
        direction: str,
        intensity: str | None = None,
        steps: int | None = None,
        cue: str | None = None,
    ) -> str:
        enqueue_head_command(self.motion_queue, direction, intensity, steps, cue)
        return ""

    def set_pose(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        x_mm: float = 0.0,
        y_mm: float = 0.0,
        z_mm: float = 0.0,
        body_yaw_deg: float | None = None,
        duration_s: float = 0.6,
        hold_s: float = 0.0,
        return_mode: str = "auto",
    ) -> str:
        enqueue_pose_command(
            self.motion_queue,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            x_mm=x_mm,
            y_mm=y_mm,
            z_mm=z_mm,
            body_yaw_deg=body_yaw_deg,
            duration_s=duration_s,
            hold_s=hold_s,
            return_mode=return_mode,
        )
        return ""

    def play_emotion(
        self,
        category: str | None = None,
        emotion_name: str | None = None,
    ) -> str:
        # Specific name takes priority
        if emotion_name and emotion_name in ALL_EMOTION_NAMES:
            name = emotion_name
        elif category and category in EMOTION_CATEGORIES:
            name = random.choice(EMOTION_CATEGORIES[category])
        else:
            return "Unknown emotion"
        enqueue_emotion_command(self.motion_queue, name)
        return ""

    # ------------------------------------------------------------------
    # Lesson flow
    # ------------------------------------------------------------------

    def next_example_question(self) -> str:
        self.firebase.log_message("system", "next example question")
        return self.firebase.get_next_example_question()

    def start_quiz(self) -> str:
        self.firebase.log_message("system", "start quiz")
        return "Quiz started"

    def end_conversation(self) -> str:
        self.firebase.log_message("system", "end conversation")
        return "Conversation ended"
