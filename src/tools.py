import asyncio
import random

from firebase_helper import FirebaseHelper
from motion import (
    enqueue_head_command, enqueue_pose_command, enqueue_emotion_command,
    enqueue_return_home_command,
    EMOTION_CATEGORIES, ALL_EMOTION_NAMES,
)


class Tools:
    def __init__(self, firebase: FirebaseHelper, motion_queue: asyncio.Queue, vision=None):
        self.firebase = firebase
        self.motion_queue = motion_queue
        self.vision = vision  # ReachyVision | None

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

    def return_home(self) -> str:
        enqueue_return_home_command(self.motion_queue)
        return ""

    async def get_face_position(self) -> str:
        """Returns where the student's face is in the camera view plus suggested head adjustment."""
        if self.vision is None:
            return "Vision not available"
        face = await self.vision.get_face_center()
        if face is None:
            return "No face detected — student may be out of frame or camera still warming up"
        u, v = face
        async with self.vision._lock:
            frame = self.vision._latest_frame_raw
        if frame is None:
            return "No frame available"
        h, w = frame.shape[:2]
        # Normalize to [-1, 1]: -1=left/top edge, 0=center, +1=right/bottom edge
        nx = (u / w - 0.5) * 2
        ny = (v / h - 0.5) * 2
        # Approximate angle to center the face (camera ~70° HFOV, ~55° VFOV)
        yaw_adjust = round(-nx * 32)   # positive=turn head left, negative=turn right
        pitch_adjust = round(ny * 25)  # positive=look down, negative=look up
        horiz = ("to your left" if nx < -0.15 else "to your right" if nx > 0.15 else None)
        vert = ("above center" if ny < -0.15 else "below center" if ny > 0.15 else None)
        pos_parts = [p for p in [horiz, vert] if p]
        if not pos_parts:
            return "Face is well-centered in your view — no adjustment needed."
        pos_desc = " and ".join(pos_parts)
        return (
            f"Face is {pos_desc}. "
            f"To center them: adjust yaw_deg by {yaw_adjust:+d} and pitch_deg by {pitch_adjust:+d} "
            f"from your current head position."
        )

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
