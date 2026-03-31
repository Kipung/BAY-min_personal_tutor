from __future__ import annotations

import asyncio
from contextlib import suppress

from vision import ReachyVision, send_vision_loop

from google import genai
from google.oauth2 import service_account
from reachy_mini import ReachyMini
from reachy_mini.motion.recorded_move import RecordedMoves

from audio_adapters import capture_mic_loop, play_speaker_loop
from bluetooth_helper import wait_for_active_user_async
from firebase_helper import FirebaseHelper
from gemini_live import receive_loop, send_mic_loop, MODEL, build_live_config
from motion import motion_worker_loop, MOTION_QUEUE_MAX


SPEAKER_QUEUE_MAX = 60
EMOTION_DATASET = "pollen-robotics/reachy-mini-emotions-library"
MIC_QUEUE_MAX = 8


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
            uid = await wait_for_active_user_async()
            firebase.set_user(uid)
            disconnected_event = asyncio.Event()
            
            while True:
                # ── STATE 2: Wait for module selection ──────────────────────
                if not firebase.module_id:
                    print("[state] State 2: Waiting for module selection...")
                    _, pending = await asyncio.wait(
                        {
                            asyncio.create_task(firebase.module_selected_event.wait(), name="module"),
                            asyncio.create_task(disconnected_event.wait(), name="disconnect"),
                        },
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()
                        with suppress(asyncio.CancelledError):
                            await t

                    if disconnected_event.is_set():
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
                            motion_queue,
                            disconnected_event, firebase.module_exited_event,
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
