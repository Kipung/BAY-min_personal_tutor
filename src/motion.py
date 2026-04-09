from __future__ import annotations

import asyncio
import math
from contextlib import suppress

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from reachy_mini.motion.recorded_move import RecordedMoves


# --- Queue & timing ---
MOTION_QUEUE_MAX = 20
MOTION_DEFAULT_DURATION = 0.6
HEAD_MOVE_DURATION_S = 0.35          # fallback; see _head_duration()
HEAD_CMD_MIN_INTERVAL_S = 0.18

# --- Joint limits ---
BASE_YAW_STEP_RAD = 0.22
BASE_YAW_MIN_RAD = -1.00
BASE_YAW_MAX_RAD = 1.00

HEAD_YAW_MIN_DEG = -35
HEAD_YAW_MAX_DEG = 35
HEAD_PITCH_MIN_DEG = -25
HEAD_PITCH_MAX_DEG = 25
HEAD_ROLL_MIN_DEG = -20
HEAD_ROLL_MAX_DEG = 20

# --- Step sizes per direction ---
YAW_STEP_DEG = 13.0
PITCH_STEP_DEG = 10.0
ROLL_STEP_DEG = 8.0

MOVE_HEAD_TOOL_DECLARATION = {
    "name": "move_head",
    "description": (
        "Move Reachy's head/base in a direction. Optionally include intensity or steps "
        "based on user wording (for example: slightly, more, all the way)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": [
                    "left", "right", "up", "down",
                    "tilt_left", "tilt_right",
                    "center", "base_left", "base_right",
                ],
            },
            "intensity": {
                "type": "string",
                "enum": ["tiny", "small", "medium", "large", "max"],
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
            },
            "cue": {
                "type": "string",
                "description": "Original wording cue for motion size, e.g. 'a bit more', 'all the way'.",
            },
        },
        "required": ["direction"],
    },
}

SET_POSE_TOOL_DECLARATION = {
    "name": "set_pose",
    "description": (
        "Set an exact head pose using angles. Use this for combined or diagonal directions "
        "that move_head can't express (e.g. 'bottom right' = pitch_deg=20, yaw_deg=-25; "
        "'top left' = pitch_deg=-20, yaw_deg=25; 'reset/center' = all zeros). "
        "Coordinate signs: yaw_deg negative=right, positive=left; "
        "pitch_deg positive=down, negative=up; roll_deg positive=tilt-left, negative=tilt-right."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "yaw_deg":      {"type": "number", "minimum": -35, "maximum": 35},
            "pitch_deg":    {"type": "number", "minimum": -25, "maximum": 25},
            "roll_deg":     {"type": "number", "minimum": -20, "maximum": 20},
            "x_mm":         {"type": "number", "minimum": -20, "maximum": 20},
            "y_mm":         {"type": "number", "minimum": -25, "maximum": 25},
            "z_mm":         {"type": "number", "minimum": -20, "maximum": 20},
            "body_yaw_deg": {"type": "number", "minimum": -60, "maximum": 60},
            "duration_s":   {"type": "number", "minimum": 0.2, "maximum": 4.0},
            "hold_s":       {"type": "number", "minimum": 0.0, "maximum": 8.0},
            "return_mode":  {"type": "string", "enum": ["auto", "keep", "neutral"]},
        },
        "required": [],
    },
}

# Curated subsets of the 85 available emotions for the LLM to choose from
EMOTION_CATEGORIES = {
    "agreement":    ["yes1", "uh_huh_tilt", "simple_nod"],
    "disagreement": ["no1", "no_excited1", "no_sad1"],
    "thinking":     ["thoughtful1", "thoughtful2"],
    "curious":      ["curious1", "inquiring1", "inquiring2", "inquiring3"],
    "happy":        ["cheerful1", "enthusiastic1", "enthusiastic2", "laughing1", "laughing2"],
    "praise":       ["proud1", "proud2", "proud3", "success1", "success2"],
    "encourage":    ["helpful1", "helpful2", "understanding1", "understanding2", "calming1"],
    "surprised":    ["surprised1", "surprised2", "amazed1"],
    "confused":     ["confused1", "uncertain1", "lost1"],
    "greeting":     ["welcoming1", "welcoming2"],
    "sad":          ["sad1", "sad2", "downcast1"],
    "attentive":    ["attentive1", "attentive2"],
    "oops":         ["oops1", "oops2"],
}

ALL_EMOTION_NAMES = sorted({name for names in EMOTION_CATEGORIES.values() for name in names})

PLAY_EMOTION_TOOL_DECLARATION = {
    "name": "play_emotion",
    "description": (
        "Play a full-body emotion animation on Reachy. Use this to express emotions naturally "
        "during conversation — nodding when agreeing, looking curious when the student asks a "
        "question, celebrating when they get an answer right, etc. "
        "Choose by category OR by specific emotion name."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": list(EMOTION_CATEGORIES.keys()),
                "description": (
                    "Pick a category and a fitting animation will be chosen. "
                    "Categories: agreement (nod/yes), disagreement (no/shake), "
                    "thinking (pondering), curious (investigating), happy (joy/laugh), "
                    "praise (proud/success), encourage (helpful/calming), "
                    "surprised (amazed), confused (uncertain), greeting (welcome), "
                    "sad, attentive (listening), oops (mistake)."
                ),
            },
            "emotion_name": {
                "type": "string",
                "description": "Specific emotion name if you want precise control. Overrides category.",
            },
        },
        "required": [],
    },
}

VALID_MOVE_DIRECTIONS = {
    "left", "right", "up", "down",
    "tilt_left", "tilt_right",
    "center", "base_left", "base_right",
}

MOVE_INTENSITY_SCALE = {
    "tiny":   0.7,
    "small":  1.0,
    "medium": 1.6,
    "large":  2.4,
    "max":    3.2,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _to_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _head_duration(delta_yaw: float, delta_pitch: float, delta_roll: float) -> float:
    """Duration that scales with angular distance — small moves snappy, large moves deliberate."""
    dist = math.sqrt(delta_yaw ** 2 + delta_pitch ** 2 + delta_roll ** 2)
    return _clamp(0.15 + dist * 0.012, 0.20, 1.2)


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

    if any(t in cue_text for t in ["all the way", "way more", "much more", "a lot", "max", "maximum", "far"]):
        scale = max(scale, 3.0)
    elif any(t in cue_text for t in ["more", "further", "bigger"]):
        scale = max(scale, 2.0)
    elif any(t in cue_text for t in ["slight", "slightly", "little", "a bit", "bit"]):
        scale = max(scale, 0.8)

    return direction, scale


# ---------------------------------------------------------------------------
# Queue helpers (called from Tools)
# ---------------------------------------------------------------------------

def enqueue_head_command(
    motion_queue: asyncio.Queue,
    direction: str,
    intensity: str | None = None,
    steps: int | float | None = None,
    cue: str | None = None,
) -> None:
    """Drop stale head commands and push the latest one."""
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

    motion_queue.put_nowait({
        "kind": "head",
        "direction": direction,
        "intensity": intensity,
        "steps": steps,
        "cue": cue,
    })


def enqueue_pose_command(
    motion_queue: asyncio.Queue,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    x_mm: float = 0.0,
    y_mm: float = 0.0,
    z_mm: float = 0.0,
    body_yaw_deg: float | None = None,
    duration_s: float = MOTION_DEFAULT_DURATION,
    hold_s: float = 0.0,
    return_mode: str = "auto",
) -> None:
    if motion_queue.full():
        with suppress(asyncio.QueueEmpty):
            motion_queue.get_nowait()

    motion_queue.put_nowait({
        "kind": "pose",
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "z_mm": z_mm,
        "body_yaw_deg": body_yaw_deg,
        "duration_s": duration_s,
        "hold_s": hold_s,
        "return_mode": return_mode,
    })


def enqueue_emotion_command(motion_queue: asyncio.Queue, name: str) -> None:
    if motion_queue.full():
        with suppress(asyncio.QueueEmpty):
            motion_queue.get_nowait()
    motion_queue.put_nowait({"kind": "emotion", "name": name})


# ---------------------------------------------------------------------------
# Worker loop (run as asyncio task from main.py)
# ---------------------------------------------------------------------------

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
            direction, step_scale = _normalize_move_direction_and_scale(
                direction_raw, cmd.get("intensity"), cmd.get("steps"), cmd.get("cue")
            )

            yaw_step   = YAW_STEP_DEG      * step_scale
            pitch_step = PITCH_STEP_DEG    * step_scale
            roll_step  = ROLL_STEP_DEG     * step_scale
            base_step  = BASE_YAW_STEP_RAD * step_scale

            now = asyncio.get_running_loop().time()
            if direction != "center" and (now - last_head_cmd_ts) < HEAD_CMD_MIN_INTERVAL_S:
                continue

            prev_yaw, prev_pitch, prev_roll = current_yaw_deg, current_pitch_deg, current_roll_deg

            if direction == "right":
                current_yaw_deg = _clamp(current_yaw_deg - yaw_step, HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
            elif direction == "left":
                current_yaw_deg = _clamp(current_yaw_deg + yaw_step, HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
            elif direction == "up":
                current_pitch_deg = _clamp(current_pitch_deg - pitch_step, HEAD_PITCH_MIN_DEG, HEAD_PITCH_MAX_DEG)
            elif direction == "down":
                current_pitch_deg = _clamp(current_pitch_deg + pitch_step, HEAD_PITCH_MIN_DEG, HEAD_PITCH_MAX_DEG)
            elif direction == "tilt_left":
                current_roll_deg = _clamp(current_roll_deg + roll_step, HEAD_ROLL_MIN_DEG, HEAD_ROLL_MAX_DEG)
            elif direction == "tilt_right":
                current_roll_deg = _clamp(current_roll_deg - roll_step, HEAD_ROLL_MIN_DEG, HEAD_ROLL_MAX_DEG)
            elif direction == "base_left":
                # Head leads: anticipate the turn by glancing left
                head_lead_yaw = _clamp(current_yaw_deg + yaw_step * 0.5, HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
                await asyncio.to_thread(
                    mini.goto_target,
                    head=create_head_pose(yaw=head_lead_yaw, pitch=current_pitch_deg, roll=current_roll_deg, degrees=True),
                    body_yaw=current_body_yaw,
                    duration=0.25,
                )
                await asyncio.sleep(0.18)
                current_body_yaw = _clamp(current_body_yaw + base_step, BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)
                current_yaw_deg = head_lead_yaw
            elif direction == "base_right":
                head_lead_yaw = _clamp(current_yaw_deg - yaw_step * 0.5, HEAD_YAW_MIN_DEG, HEAD_YAW_MAX_DEG)
                await asyncio.to_thread(
                    mini.goto_target,
                    head=create_head_pose(yaw=head_lead_yaw, pitch=current_pitch_deg, roll=current_roll_deg, degrees=True),
                    body_yaw=current_body_yaw,
                    duration=0.25,
                )
                await asyncio.sleep(0.18)
                current_body_yaw = _clamp(current_body_yaw - base_step, BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)
                current_yaw_deg = head_lead_yaw
            elif direction == "center":
                current_yaw_deg, current_pitch_deg, current_roll_deg = 0.0, 0.0, 0.0

            dur = _head_duration(
                current_yaw_deg - prev_yaw,
                current_pitch_deg - prev_pitch,
                current_roll_deg - prev_roll,
            )

            await asyncio.to_thread(
                mini.goto_target,
                head=create_head_pose(
                    yaw=current_yaw_deg,
                    pitch=current_pitch_deg,
                    roll=current_roll_deg,
                    degrees=True,
                ),
                body_yaw=current_body_yaw,
                duration=dur,
            )
            last_head_cmd_ts = now


        elif kind == "pose":
            yaw   = _clamp(_to_float(cmd.get("yaw_deg",   0.0), 0.0), HEAD_YAW_MIN_DEG,   HEAD_YAW_MAX_DEG)
            pitch = _clamp(_to_float(cmd.get("pitch_deg", 0.0), 0.0), HEAD_PITCH_MIN_DEG, HEAD_PITCH_MAX_DEG)
            roll  = _clamp(_to_float(cmd.get("roll_deg",  0.0), 0.0), HEAD_ROLL_MIN_DEG,  HEAD_ROLL_MAX_DEG)
            x_mm  = _to_float(cmd.get("x_mm", 0.0), 0.0)
            y_mm  = _to_float(cmd.get("y_mm", 0.0), 0.0)
            z_mm  = _to_float(cmd.get("z_mm", 0.0), 0.0)

            body_yaw_deg = cmd.get("body_yaw_deg")
            new_body_yaw = current_body_yaw
            if body_yaw_deg is not None:
                new_body_yaw = _clamp(math.radians(float(body_yaw_deg)), BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)

            current_yaw_deg, current_pitch_deg, current_roll_deg = yaw, pitch, roll

            duration_s  = float(cmd.get("duration_s", MOTION_DEFAULT_DURATION))
            hold_s      = float(cmd.get("hold_s", 0.0))
            return_mode = str(cmd.get("return_mode", "auto"))

            # Head leads body: move head first, then body catches up
            if new_body_yaw != current_body_yaw:
                await asyncio.to_thread(
                    mini.goto_target,
                    head=create_head_pose(
                        x=x_mm, y=y_mm, z=z_mm,
                        roll=current_roll_deg, pitch=current_pitch_deg, yaw=current_yaw_deg,
                        degrees=True, mm=True,
                    ),
                    body_yaw=current_body_yaw,  # body stays while head moves first
                    duration=duration_s * 0.5,
                )
                await asyncio.sleep(0.18)
                current_body_yaw = new_body_yaw
                await asyncio.to_thread(
                    mini.goto_target,
                    head=create_head_pose(
                        x=x_mm, y=y_mm, z=z_mm,
                        roll=current_roll_deg, pitch=current_pitch_deg, yaw=current_yaw_deg,
                        degrees=True, mm=True,
                    ),
                    body_yaw=current_body_yaw,
                    duration=duration_s * 0.6,
                )
            else:
                await asyncio.to_thread(
                    mini.goto_target,
                    head=create_head_pose(
                        x=x_mm, y=y_mm, z=z_mm,
                        roll=current_roll_deg, pitch=current_pitch_deg, yaw=current_yaw_deg,
                        degrees=True, mm=True,
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
                    await mini.async_play_move(emotions.get(name), sound=False)


# ---------------------------------------------------------------------------
# Idle behavior loop  (runs as a separate asyncio task)
