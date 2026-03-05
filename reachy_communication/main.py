from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime

from google import genai
from google.oauth2 import service_account
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose

import firebase_admin
from firebase_admin import credentials, firestore

from audio_adapters import (
    PCMFramer,
    resample_from_24kHz,
    resample_to_16k_mono
)
MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"
LIVE_CONFIG = {
    "response_modalities": ["AUDIO"],
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
                "description": "Move Reachy's head in a direction. Argument should be one of: left, right, center, up.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["left", "right", "center", "up"]
                        }
                    },
                    "required": ["direction"]
                }
            }
        ]
    }]
}
SPEAKER_QUEUE_MAX = 60
MIC_QUEUE_MAX = 6
PLAY_CHUNK_SECONDS = 0.02


def clear_queue(q: asyncio.Queue) -> None:
    while True:
        with suppress(asyncio.QueueEmpty):
            q.get_nowait()
            continue
        break


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
            drop_oldest_put_nowait(mic_queue, frame)
        await asyncio.sleep(0)


async def send_mic_loop(session, mic_queue: asyncio.Queue) -> None:
    """
    Read PCM16 16kHz mono frames from mic_queue and send to Gemini Live as realtime input.
    """
    while True:
        frame = await mic_queue.get()
        await session.send_realtime_input(
            audio={"data": frame, "mime_type": "audio/pcm"}
        )


async def receive_loop(
    session,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
    mini: ReachyMini,
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
                elif call.name == "move_head" and call.args and isinstance(call.args, dict):
                    direction = call.args.get("direction")
                    if isinstance(direction, str):
                        move_head(mini, direction)
                        await session.send_tool_response(
                            function_responses=[{
                                "name": "move_head",
                                "response": {"result": f"Moved head {direction}"},
                                "id": call.id,
                            }]
                        )
                break
            
            if sc is None:
                continue
                
            if sc.interrupted and not interrupted_event.is_set():
                mini.media.stop_playing()
                interrupted_event.set()
                clear_queue(speaker_queue)
                print("[live] generation interrupted -> clearing speaker queue and waiting for new audio")
                break

            if sc.input_transcription:
                user_tx = sc.input_transcription.text
                if user_tx and user_tx.strip():
                    #placeholder for uploading to firestore
                    print(f"USER: {user_tx}")
                    message_collection.add({"from": "user", "message": user_tx, "createdAt": datetime.now()})
            
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


def move_head(mini: ReachyMini, direction: str):
    """
    Example function for a Gemini Live tool call to move Reachy's head.
    """
    movement_map = {
        "left": create_head_pose(y=20, mm=True),
        "right": create_head_pose(y=-20, mm=True),
        "center": create_head_pose(y=0, z=0, mm=True),
        "up": create_head_pose(z=20, mm=True)
    }
    if direction in movement_map:
        mini.goto_target(head=movement_map[direction], duration=0.5)

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

            print("Connected to Gemini Live, starting communication loop...")
            input("press enter to start")
            tasks = [
                asyncio.create_task(capture_mic_loop(mini, mic_queue), name="capture_mic"),
                asyncio.create_task(send_mic_loop(session, mic_queue), name="send_mic"),
                asyncio.create_task(play_speaker_loop(mini, speaker_queue, interrupted_event), name="play_speaker"),
            ]
            try:
                await receive_loop(session, speaker_queue, interrupted_event, mini)
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
