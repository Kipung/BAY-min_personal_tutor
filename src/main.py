from __future__ import annotations

import asyncio
from contextlib import suppress

from vision import ReachyVision, send_vision_loop
from rag import FirestoreRAG

from google import genai
from google.oauth2 import service_account
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

from audio_adapters import capture_mic_loop, play_speaker_loop
from bluetooth_helper import wait_for_active_user
from firebase_helper import FirebaseHelper
from gemini_live import receive_loop, send_mic_loop, MODEL
from motion import (
    motion_worker_loop,
    MOTION_QUEUE_MAX,
    MOVE_HEAD_TOOL_DECLARATION,
    SET_POSE_TOOL_DECLARATION,
)


def build_live_config(lesson_context: str = "") -> dict:
    """Build the Gemini Live session config, optionally injecting RAG lesson context."""
    base_instruction = (
        "You are BAY-min, a friendly and encouraging 4th-grade math tutor robot. "
        "Always respond in English only. If the student speaks another language, gently continue in English. "
        "You will periodically receive images from your front-facing camera — use them to "
        "describe what you see, answer visual questions, or react to the student's environment. "
        "For movement requests, map wording cues to motion size: words like 'slightly'/'a bit' -> "
        "small move, 'more'/'further' -> medium move, and 'way more'/'a lot'/'all the way' -> large move. "
        "Keep explanations simple, positive, and age-appropriate for a 4th grader. "
        "Always give ONE response per student turn, then wait for them to reply before continuing."
    )

    if lesson_context:
        system_instruction = (
            base_instruction
            + "\n\n"
            + lesson_context
        )
    else:
        system_instruction = base_instruction

    return {
        "response_modalities": ["AUDIO"],
        "system_instruction": system_instruction,
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
                MOVE_HEAD_TOOL_DECLARATION,
                SET_POSE_TOOL_DECLARATION,
                {
                    "name": "next_example_question",
                    "description": "move on to the next example question in the current module. No arguments. Returns the question, answer, and steps to walk through to get to the answer."
                },
                {
                    "name": "start_quiz",
                    "description": "Start the module's quiz. No arguments."
                }
            ]
        }]
    }


SPEAKER_QUEUE_MAX = 60
EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"
MIC_QUEUE_MAX = 8


async def run() -> None:
    user_id = wait_for_active_user()

    firebase = FirebaseHelper()
    firebase.set_user(user_id)

    with ReachyMini() as mini:
        mini.media.start_recording()
        mini.media.start_playing()

        rag = FirestoreRAG(firebase.db, module_pattern=r"^math_grade4_ch1_les\d+_")
        rag.load()
        lesson_context = rag.build_system_context()

        live_config = build_live_config(lesson_context)

        vision = ReachyVision(mini)

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
        async with client.aio.live.connect(model=MODEL, config=live_config) as session:
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
                asyncio.create_task(motion_worker_loop(mini, motion_queue, interrupted_event, emotions), name="motion_worker"),
                asyncio.create_task(vision.capture_loop(), name="capture_vision"),
                asyncio.create_task(send_vision_loop(session, vision), name="send_vision"),
            ]
            try:
                await receive_loop(session, speaker_queue, interrupted_event, mini, firebase, motion_queue)
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
