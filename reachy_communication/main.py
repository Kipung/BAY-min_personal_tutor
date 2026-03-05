from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime

import math

from google import genai
from google.oauth2 import service_account
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

import firebase_admin
from firebase_admin import credentials, firestore


from reachy_mini.motion.recorded_move import RecordedMoves

from audio_adapters import (
    PCMFramer,
    resample_from_24kHz,
    resample_to_16k_mono
)
MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"
LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],
    "system_instruction": "Always respond in English only. If the user speaks another language, continue in English. For movement requests, map wording cues to motion size: words like 'slightly'/'a bit' -> small move, 'more'/'further' -> medium move, and 'way more'/'a lot'/'all the way' -> large move.",
    "input_audio_transcription": {},
    "output_audio_transcription": {},
    "speech_config": {
        "language_code": "en-US",
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Fenrir"}}
    },
    "tools": [{
        "function_declarations": [
            {
                "name": "end_conversation",
                "description": "End the conversation. Gemini Live will stop generating and close the session after calling this."
            },
            {
                "name": "move_head",
                "description": "Move Reachy's head/base in a direction. Optionally include intensity or steps based on user wording (for example: slightly, more, all the way).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": [
                                "left", "right", "up", "down",
                                "tilt_left", "tilt_right",
                                "center", "base_left", "base_right"
                            ]
                        },
                        "intensity": {
                            "type": "string",
                            "enum": ["tiny", "small", "medium", "large", "max"]
                        },
                        "steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 4
                        },
                        "cue": {
                            "type": "string",
                            "description": "Original wording cue for motion size, e.g. 'a bit more', 'all the way'."
                        }
                    },
                    "required": ["direction"]
                }
            },
            {
                "name": "set_pose",
                "description": "Set combined head/base pose with optional hold and return behavior.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "yaw_deg": {"type": "number", "minimum": -35, "maximum": 35},
                        "pitch_deg": {"type": "number", "minimum": -25, "maximum": 25},
                        "roll_deg": {"type": "number", "minimum": -20, "maximum": 20},
                        "x_mm": {"type": "number", "minimum": -20, "maximum": 20},
                        "y_mm": {"type": "number", "minimum": -25, "maximum": 25},
                        "z_mm": {"type": "number", "minimum": -20, "maximum": 20},
                        "body_yaw_deg": {"type": "number", "minimum": -60, "maximum": 60},
                        "duration_s": {"type": "number", "minimum": 0.2, "maximum": 4.0},
                        "hold_s": {"type": "number", "minimum": 0.0, "maximum": 8.0},
                        "return_mode": {"type": "string", "enum": ["auto", "keep", "neutral"]}
                    },
                    "required": []
                }
            }
        ]
    }]
}
SPEAKER_QUEUE_MAX = 60
MOTION_QUEUE_MAX = 20
MOTION_DEFAULT_DURATION = 0.6
EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"
MIC_QUEUE_MAX = 8
PLAY_CHUNK_SECONDS = 0.02
MIC_PREROLL_FRAMES = 10  # number of initial mic frames to skip to avoid stale audio


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


def clear_queue(q: asyncio.Queue) -> None:
    while True:
        with suppress(asyncio.QueueEmpty):
            q.get_nowait()
            continue
        break


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

def drop_oldest_put_nowait(q: asyncio.Queue, item: bytes) -> None:
    if q.full():
        with suppress(asyncio.QueueEmpty):
            q.get_nowait()
    q.put_nowait(item)


async def capture_mic_loop(mini: ReachyMini, mic_queue: asyncio.Queue) -> None:
    """
    Capture audio from Reachy, convert to PCM16 16kHz mono, and push to mic_queue in 20ms frames.
    """
    framer = PCMFramer()
    input_sr = mini.media.get_input_audio_samplerate()
    input_ch = mini.media.get_input_channels()
    while True:
        audio = mini.media.get_audio_sample()
        if audio is None:
            await asyncio.sleep(0.002)
            continue

        pcm16_16k = resample_to_16k_mono(audio, input_sr, input_ch)
        framer.push(pcm16_16k)

        for frame in framer.pop_frames():
            # drop_oldest_put_nowait(mic_queue, frame)
            await mic_queue.put(frame)
        await asyncio.sleep(0)


async def send_mic_loop(session, mic_queue: asyncio.Queue) -> None:
    """
    Read PCM16 16kHz mono frames from mic_queue and send to Gemini Live as realtime input.
    """
    buffered_frames = [] # count frames sent before conversation starts, to skip initial stale audio
    started = False

    while True:
        frame = await mic_queue.get()
        if not started:
            buffered_frames.append(frame)
            if len(buffered_frames) < MIC_PREROLL_FRAMES:
                continue

            for buffered_frame in buffered_frames:
                await session.send_realtime_input(
                    audio={"data": buffered_frame, "mime_type": "audio/pcm"}
                )
            buffered_frames.clear()
            started = True
            continue

        await session.send_realtime_input(
            audio={"data": frame, "mime_type": "audio/pcm"}
        )


async def receive_loop(
    session,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
    mini: ReachyMini,
    motion_queue: asyncio.Queue,
) -> None:
    """
    Receive Gemini Live responses, extract audio and text, and push audio to speaker_queue.
    """
    file = open("gemini_live_responses.txt", "w", encoding="utf-8")
    ended = False
    
    creds = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(creds)
    db = firestore.client()
    message_collection = db.collection("conversations").document("BEYAvvfuXVZYo4lLPE5KFKLakId2").collection("messages")
    while not ended:
        reachy_response_text = ""
        async for response in session.receive():
            file.write(str(response) + f"\n{'-'*20}\n")
            sc = response.server_content
            
            for call in response.tool_call.function_calls if response.tool_call else []:
                print(f"TOOL CALL: {call}")

                if call.name == "end_conversation":
                    ended = True
                    continue

                if call.name == "move_head" and call.args and isinstance(call.args, dict):
                    direction = call.args.get("direction")
                    if isinstance(direction, str):
                        intensity = call.args.get("intensity")
                        steps = call.args.get("steps")
                        cue = call.args.get("cue")
                        enqueue_latest_head_command(
                            motion_queue,
                            direction,
                            intensity=intensity if isinstance(intensity, str) else None,
                            steps=steps,
                            cue=cue if isinstance(cue, str) else None,
                        )
                        tool_result = {
                            "result": f"Queued head motion: {direction}",
                            "intensity": intensity,
                            "steps": steps,
                        }
                    else:
                        tool_result = {"error": "Invalid direction"}

                    await session.send_tool_response(
                        function_responses=[{
                            "name": "move_head",
                            "response": tool_result,
                            "id": call.id,
                        }]
                    )
                if call.name == "set_pose" and call.args and isinstance(call.args, dict):
                    if motion_queue.full():
                        with suppress(asyncio.QueueEmpty):
                            motion_queue.get_nowait()

                    motion_queue.put_nowait({
                        "kind": "pose",
                        "yaw_deg": call.args.get("yaw_deg", 0.0),
                        "pitch_deg": call.args.get("pitch_deg", 0.0),
                        "roll_deg": call.args.get("roll_deg", 0.0),
                        "x_mm": call.args.get("x_mm", 0.0),
                        "y_mm": call.args.get("y_mm", 0.0),
                        "z_mm": call.args.get("z_mm", 0.0),
                        "body_yaw_deg": call.args.get("body_yaw_deg"),
                        "duration_s": call.args.get("duration_s", MOTION_DEFAULT_DURATION),
                        "hold_s": call.args.get("hold_s", 0.0),
                        "return_mode": call.args.get("return_mode", "auto"),
                    })

                    await session.send_tool_response(
                        function_responses=[{
                            "name": "set_pose",
                            "response": {"result": "Queued combined pose"},
                            "id": call.id,
                        }]
                    )

            if sc is None:
                continue
                
            if sc.interrupted and not interrupted_event.is_set():
                mini.media.stop_playing()
                interrupted_event.set()


                clear_queue(speaker_queue)
                clear_queue(motion_queue)
                print("[live] generation interrupted -> cleared speaker + motion queues")
                break

            if sc.input_transcription:
                user_tx = sc.input_transcription.text
                if user_tx and user_tx.strip():
                    #placeholder for uploading to firestore
                    print(f"USER: {user_tx}")
                    message_collection.add({"from": "student", "message": user_tx, "createdAt": datetime.now()})
            
            if sc.output_transcription:
                spoken_tx = sc.output_transcription.text
                if spoken_tx and spoken_tx.strip():
                    print(f"ASSISTANT (partial): {spoken_tx}")
                    reachy_response_text += spoken_tx
            
            audio_chunks = sc.model_turn.parts if sc.model_turn and sc.model_turn.parts else []
            for part in audio_chunks:
                inline_data = part.inline_data
                data = inline_data.data if inline_data else None
                if isinstance(data, (bytes, bytearray)):
                    drop_oldest_put_nowait(speaker_queue, bytes(data))

        if interrupted_event.is_set():
            interrupted_event.clear()
            mini.media.start_playing()
            print("[live] generation complete after interruption -> ready to receive new assistant audio")
            reachy_response_text += " [generation interrupted]"
        #placeholder for uploading to firestore
        if not ended:
            print(f"ASSISTANT FINAL: {reachy_response_text}")
            message_collection.add({"from": "reachy", "message": reachy_response_text, "createdAt": datetime.now()})
    file.close()


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def move_head(mini: ReachyMini, direction: str, current_body_yaw: float) -> float:
    head_map = {
        "left": create_head_pose(yaw=25, degrees=True),
        "right": create_head_pose(yaw=-25, degrees=True),
        "up": create_head_pose(pitch=-15, degrees=True),
        "down": create_head_pose(pitch=15, degrees=True),
        "tilt_left": create_head_pose(roll=12, degrees=True),
        "tilt_right": create_head_pose(roll=-12, degrees=True),
        "center": create_head_pose(),
        "base_left": create_head_pose(),
        "base_right": create_head_pose(),
    }

    if direction not in head_map:
        return current_body_yaw

    target_body_yaw = current_body_yaw
    if direction == "base_left":
        target_body_yaw = _clamp(current_body_yaw + BASE_YAW_STEP_RAD, BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)
    elif direction == "base_right":
        target_body_yaw = _clamp(current_body_yaw - BASE_YAW_STEP_RAD, BASE_YAW_MIN_RAD, BASE_YAW_MAX_RAD)

    mini.goto_target(
        head=head_map[direction],
        body_yaw=target_body_yaw,
        duration=0.7,
    )
    return target_body_yaw

async def play_speaker_loop(
    mini: ReachyMini,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
) -> None:
    """
    Read PCM16 24kHz mono audio chunks from speaker_queue, convert to Reachy format, and play.
    """
    output_sr = mini.media.get_output_audio_samplerate()
    slice_n = int(output_sr * PLAY_CHUNK_SECONDS)
    while True:
        audio_24k_pcm16 = await speaker_queue.get()

        if interrupted_event.is_set():
            continue

        out = resample_from_24kHz(audio_24k_pcm16, output_sr)
        for start in range(0, out.shape[0], slice_n):
            if interrupted_event.is_set():
                break
            mini.media.push_audio_sample(out[start : start + slice_n])
            await asyncio.sleep(0)


async def run() -> None:
    with ReachyMini() as mini:
        mini.media.start_recording()
        mini.media.start_playing()

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
        async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
            mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
            speaker_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=SPEAKER_QUEUE_MAX)
            interrupted_event = asyncio.Event()
            motion_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=MOTION_QUEUE_MAX)
            emotions = RecordedMoves(EMOTION_DATASET)

            print("Connected to Gemini Live, starting communication loop...")
            input("press enter to start")
            tasks = [
                asyncio.create_task(capture_mic_loop(mini, mic_queue), name="capture_mic"),
                asyncio.create_task(send_mic_loop(session, mic_queue), name="send_mic"),
                asyncio.create_task(play_speaker_loop(mini, speaker_queue, interrupted_event), name="play_speaker"),
                asyncio.create_task(
                    motion_worker_loop(mini, motion_queue, interrupted_event, emotions),
                    name="motion_worker",
),
            ]
            try:
                await receive_loop(session, speaker_queue, interrupted_event, mini, motion_queue)
            finally:
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with suppress(asyncio.CancelledError):
                        await task
                mini.media.stop_recording()
                mini.media.stop_playing()

def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
