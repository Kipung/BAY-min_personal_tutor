from __future__ import annotations

import asyncio
from contextlib import suppress

from google import genai
from google.oauth2 import service_account
from reachy_mini import ReachyMini

from audio_adapters import capture_mic_loop, play_speaker_loop
from bluetooth_helper import wait_for_active_user
from firebase_helper import FirebaseHelper
from gemini_live import receive_loop, send_mic_loop, LIVE_CONFIG, MODEL


SPEAKER_QUEUE_MAX = 60
MIC_QUEUE_MAX = 8


async def run() -> None:
    user_id = wait_for_active_user()

    firebase = FirebaseHelper()
    firebase.set_user(user_id)

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

            print("Connected to Gemini Live, starting communication loop...")
            input("press enter to start")
        
            tasks = [
                asyncio.create_task(capture_mic_loop(mini, mic_queue), name="capture_mic"),
                asyncio.create_task(send_mic_loop(session, mic_queue), name="send_mic"),
                asyncio.create_task(play_speaker_loop(mini, speaker_queue, interrupted_event), name="play_speaker"),
            ]
            try:
                await receive_loop(session, speaker_queue, interrupted_event, mini, firebase)
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
