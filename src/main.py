from __future__ import annotations

import asyncio
from contextlib import suppress

from vision import ReachyVision, send_vision_loop
from rag import FirestoreRAG

import math

from google import genai
from google.oauth2 import service_account
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMoves

from audio_adapters import capture_mic_loop, play_speaker_loop
from bluetooth_helper import wait_for_active_user_async
from firebase_helper import FirebaseHelper
from gemini_live import receive_loop, send_mic_loop, MODEL, build_live_config


SPEAKER_QUEUE_MAX = 60
MOTION_QUEUE_MAX = 20
MOTION_DEFAULT_DURATION = 0.6
EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"
MIC_QUEUE_MAX = 8

BASE_YAW_STEP_RAD = 0.22
BASE_YAW_MIN_RAD = -1.00
BASE_YAW_MAX_RAD = 1.00

HEAD_YAW_MIN_DEG = -35
HEAD_YAW_MAX_DEG = 35
HEAD_PITCH_MIN_DEG = -25
HEAD_PITCH_MAX_DEG = 25
HEAD_ROLL_MIN_DEG = -20
HEAD_ROLL_MAX_DEG = 20

YAW_RIGHT_STEP_DEG = -13.0
PITCH_STEP_DEG = 10.0
ROLL_STEP_DEG = 8.0
HEAD_MOVE_DURATION_S = 0.35
HEAD_CMD_MIN_INTERVAL_S = 0.18

VALID_MOVE_DIRECTIONS = {
    "left", "right", "up", "down",
    "tilt_left", "tilt_right",
    "center", "base_left", "base_right",
}

MOVE_INTENSITY_SCALE = {
    "tiny": 0.7,
    "small": 1.0,
    "medium": 1.6,
    "large": 2.4,
    "max": 3.2,
}


def enqueue_latest_head_command(
    motion_queue: asyncio.Queue,
    direction: str,
    intensity: str | None = None,
    steps: int | float | None = None,
    cue: str | None = None,
) -> None:
    buffered: list[dict] = []
    while True:
        try:
            item = motion_queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        else:
            if isinstance(item, dict):
                buffered.append(item)

    for item in buffered:
        if item.get("kind") != "head":
            motion_queue.put_nowait(item)

    if motion_queue.full():
        with suppress(asyncio.QueueEmpty):
            motion_queue.get_nowait()

    motion_queue.put_nowait(
        {
            "kind": "head",
            "direction": direction,
            "intensity": intensity,
            "steps": steps,
            "cue": cue,
        }
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_move_direction_and_scale(
    direction_raw: str,
    intensity: str | None,
    steps: int | float | None,
    cue: str | None,
) -> tuple[str, float]:
    direction = str(direction_raw).strip().lower().replace("-", "_").replace(" ", "_")
    cue_text = f"{direction_raw} {cue or ''}".lower()

    if direction not in VALID_MOVE_DIRECTIONS:
        if "base left" in cue_text or "turn left" in cue_text:
            direction = "base_left"
        elif "base right" in cue_text or "turn right" in cue_text:
            direction = "base_right"
        elif "tilt left" in cue_text:
            direction = "tilt_left"
        elif "tilt right" in cue_text:
            direction = "tilt_right"
        elif "left" in cue_text:
            direction = "left"
        elif "right" in cue_text:
            direction = "right"
        elif "up" in cue_text:
            direction = "up"
        elif "down" in cue_text:
            direction = "down"
        else:
            direction = "center"

    scale = 1.0
    intensity_key = str(intensity).strip().lower() if isinstance(intensity, str) else ""
    if intensity_key in MOVE_INTENSITY_SCALE:
        scale = MOVE_INTENSITY_SCALE[intensity_key]

    if steps is not None:
        scale = max(scale, _clamp(_to_float(steps, 1.0), 1.0, 4.0))

    if any(token in cue_text for token in ["all the way", "way more", "much more", "a lot", "max", "maximum", "far"]):
        scale = max(scale, 3.0)
    elif any(token in cue_text for token in ["more", "further", "bigger"]):
        scale = max(scale, 2.0)
    elif any(token in cue_text for token in ["slight", "slightly", "little", "a bit", "bit"]):
        scale = max(scale, 0.8)

    return direction, scale


async def motion_worker_loop(
    mini: ReachyMini,
    motion_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
    emotions: RecordedMoves,
) -> None:
    current_yaw_deg = 0.0
    current_pitch_deg = 0.0
    current_roll_deg = 0.0
    current_body_yaw = 0.0
    last_head_cmd_ts = 0.0

    while True:
        cmd = await motion_queue.get()
        if interrupted_event.is_set() or not isinstance(cmd, dict):
            continue

        kind = cmd.get("kind")

        if kind == "head":
            direction_raw = str(cmd.get("direction", "center"))
            intensity = cmd.get("intensity")
            steps = cmd.get("steps")
            cue = cmd.get("cue")
            direction, step_scale = _normalize_move_direction_and_scale(direction_raw, intensity, steps, cue)

            yaw_step = YAW_RIGHT_STEP_DEG * step_scale
            pitch_step = PITCH_STEP_DEG * step_scale
            roll_step = ROLL_STEP_DEG * step_scale
            base_step = BASE_YAW_STEP_RAD * step_scale
            now = asyncio.get_running_loop().time()

            if direction != "center" and (now - last_head_cmd_ts) < HEAD_CMD_MIN_INTERVAL_S:
                continue

            if direction == "right":
                current_yaw_deg = _clamp(current_yaw_deg + yaw_step, HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
            elif direction == "left":
                current_yaw_deg = _clamp(current_yaw_deg - yaw_step, HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
            elif direction == "up":
                current_pitch_deg = _clamp(current_pitch_deg - pitch_step, HEAD_PITCH_MIN_DEG, HEAD_PITCH_MAX_DEG)
            elif direction == "down":
                current_pitch_deg = _clamp(current_pitch_deg + pitch_step, HEAD_PITCH_MIN_DEG, HEAD_PITCH_MAX_DEG)
            elif direction == "tilt_left":
                current_roll_deg = _clamp(current_roll_deg + roll_step, HEAD_ROLL_MIN_DEG, HEAD_ROLL_MAX_DEG)
            elif direction == "tilt_right":
                current_roll_deg = _clamp(current_roll_deg - roll_step, HEAD_ROLL_MIN_DEG, HEAD_ROLL_MAX_DEG)
            elif direction == "base_left":
                current_body_yaw = _clamp(current_body_yaw + base_step, BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)
            elif direction == "base_right":
                current_body_yaw = _clamp(current_body_yaw - base_step, BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)
            elif direction == "center":
                current_yaw_deg, current_pitch_deg, current_roll_deg = 0.0, 0.0, 0.0

            await asyncio.to_thread(
                mini.goto_target,
                head=create_head_pose(
                    yaw=current_yaw_deg,
                    pitch=current_pitch_deg,
                    roll=current_roll_deg,
                    degrees=True,
                ),
                body_yaw=current_body_yaw,
                duration=HEAD_MOVE_DURATION_S,
            )
            last_head_cmd_ts = now

        elif kind == "pose":
            yaw = _clamp(_to_float(cmd.get("yaw_deg", 0.0), 0.0), HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
            pitch = _clamp(_to_float(cmd.get("pitch_deg", 0.0), 0.0), HEAD_PITCH_MIN_DEG, HEAD_PITCH_MAX_DEG)
            roll = _clamp(_to_float(cmd.get("roll_deg", 0.0), 0.0), HEAD_ROLL_MIN_DEG, HEAD_ROLL_MAX_DEG)
            x_mm = _to_float(cmd.get("x_mm", 0.0), 0.0)
            y_mm = _to_float(cmd.get("y_mm", 0.0), 0.0)
            z_mm = _to_float(cmd.get("z_mm", 0.0), 0.0)

            body_yaw_deg = cmd.get("body_yaw_deg")
            if body_yaw_deg is not None:
                current_body_yaw = _clamp(math.radians(float(body_yaw_deg)), BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)

            current_yaw_deg, current_pitch_deg, current_roll_deg = yaw, pitch, roll

            duration_s = float(cmd.get("duration_s", MOTION_DEFAULT_DURATION))
            hold_s = float(cmd.get("hold_s", 0.0))
            return_mode = str(cmd.get("return_mode", "auto"))

            await asyncio.to_thread(
                mini.goto_target,
                head=create_head_pose(
                    x=x_mm, y=y_mm, z=z_mm,
                    roll=current_roll_deg, pitch=current_pitch_deg, yaw=current_yaw_deg,
                    degrees=True, mm=True
                ),
                body_yaw=current_body_yaw,
                duration=duration_s,
            )

            if hold_s > 0:
                await asyncio.sleep(hold_s)

            if return_mode == "neutral" or (return_mode == "auto" and hold_s <= 0):
                current_yaw_deg, current_pitch_deg, current_roll_deg = 0.0, 0.0, 0.0
                await asyncio.to_thread(
                    mini.goto_target,
                    head=create_head_pose(),
                    body_yaw=current_body_yaw,
                    duration=0.6,
                )

        elif kind == "emotion":
            name = cmd.get("name")
            if isinstance(name, str):
                with suppress(Exception):
                    await mini.async_play_move(emotions.get(name))


async def run() -> None:
    firebase = FirebaseHelper()
    firebase.set_loop(asyncio.get_running_loop())

    creds = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    client = genai.Client(
        credentials=creds,
        project=creds.project_id,
        location="us-central1",
        vertexai=True,
    )
    
    with ReachyMini() as mini:
        emotions = RecordedMoves(EMOTION_DATASET)
        vision = ReachyVision(mini)

        while True:
            # ── STATE 1: Wait for Bluetooth connection ──────────────────────
            user_id = await wait_for_active_user_async()
            firebase.set_user(user_id)

            while True:
                # ── STATE 2: Wait for module selection ──────────────────────
                if not firebase.module_id:
                    print("[state] State 2: Waiting for module selection...")
                    _, pending = await asyncio.wait(
                        {
                            asyncio.create_task(firebase.module_selected_event.wait(), name="module"),
                            asyncio.create_task(firebase.disconnected_event.wait(), name="disconnect"),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                        with suppress(asyncio.CancelledError):
                            await t

                    if firebase.disconnected_event.is_set():
                        firebase.reset()
                        break  # → State 1

                # ── STATE 3: Module active — run Gemini loops ───────────────
                print(f"[state] State 3: Module '{firebase.module_id}' active.")
                mini.media.start_recording()
                mini.media.start_playing()
                
                lesson_data = firebase.get_lesson_data()
                live_config = build_live_config(lesson_data)

                async with client.aio.live.connect(model=MODEL, config=live_config) as session:
                    mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
                    speaker_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=SPEAKER_QUEUE_MAX)
                    interrupted_event = asyncio.Event()
                    motion_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=MOTION_QUEUE_MAX)

                    tasks = [
                        asyncio.create_task(capture_mic_loop(mini, mic_queue), name="capture_mic"),
                        asyncio.create_task(send_mic_loop(session, mic_queue), name="send_mic"),
                        asyncio.create_task(play_speaker_loop(mini, speaker_queue, interrupted_event), name="play_speaker"),
                        asyncio.create_task(motion_worker_loop(mini, motion_queue, interrupted_event, emotions), name="motion_worker"),
                        asyncio.create_task(vision.capture_loop(), name="capture_vision"),
                        asyncio.create_task(send_vision_loop(session, vision), name="send_vision"),
                    ]
                    try:
                        outcome = await receive_loop(
                            session, speaker_queue, interrupted_event, mini, firebase,
                            firebase.disconnected_event, firebase.module_exited_event,
                        )
                    finally:
                        for task in tasks:
                            task.cancel()
                        for task in tasks:
                            with suppress(asyncio.CancelledError):
                                await task
                        mini.media.stop_recording()
                        mini.media.stop_playing()

                print(f"[state] Session ended: {outcome}")

                if outcome == "disconnected":
                    firebase.reset()
                    break  # → State 1

                # "module_exited" or "ended" → back to State 2
                firebase.module_id = None
                firebase.module_selected_event.clear()
                firebase.module_exited_event.clear()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
