from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress

from google import genai
from google.genai.types import (
    LiveConnectConfig,
    AudioTranscriptionConfig,
    Modality,
    Tool,
    FunctionDeclaration,
    SpeechConfig,
    ProactivityConfig,
    Content,
    Part
)
from google.oauth2 import service_account
from reachy_mini import ReachyMini

from audio_adapters import (
    PCMFramer,
    resample_from_24kHz,
    resample_to_16k_mono
)
from gemini_live import (
    extract_audio_chunks,
    extract_input_transcript,
    extract_output_transcript,
    is_interrupted
)
MODEL = "gemini-live-2.5-flash-preview-native-audio-09-2025"
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
            audio={"data": frame, "mime_type": "audio/pcm;rate=16000"}
        )


async def receive_loop(
    session,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
) -> None:
    """
    Receive Gemini Live responses, extract audio and text, and push audio to speaker_queue.
    """
    file = open("gemini_live_responses.txt", "w", encoding="utf-8")
    ended = False
    while not ended:
        text = ""
        async for response in session.receive():
            file.write(str(response) + f"\n{'-'*20}\n")
            if is_interrupted(response):
                interrupted_event.set()
                clear_queue(speaker_queue)
                print("[live] interrupted=true -> dropped queued assistant audio")
                continue
                
            user_tx = extract_input_transcript(response)
            if user_tx:
                #placeholder for uploading to firestore
                print(f"USER: {user_tx}")


            spoken_tx = extract_output_transcript(response)
            if spoken_tx:
                print(f"ASSISTANT (partial): {spoken_tx}")
                text += spoken_tx

            audio_chunks = extract_audio_chunks(response)
            if audio_chunks:
                interrupted_event.clear()
                for chunk in audio_chunks:
                    drop_oldest_put_nowait(speaker_queue, chunk)

            if response.tool_call:
                for call in response.tool_call:
                    print(f"TOOL CALL: {call}")
                    ended = True
                    break

        #placeholder for uploading to firestore
        print(f"ASSISTANT FINAL: {text}")
    file.close()

# tool call
def end_conversation():
    """
    End the Gemini Live conversation, if needed.
    """
    return


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

        stereo_44k1 = resample_from_24kHz(audio_24k_pcm16, output_sr)
        for start in range(0, stereo_44k1.shape[0], slice_n):
            if interrupted_event.is_set():
                break
            mini.media.push_audio_sample(stereo_44k1[start : start + slice_n])
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
        config = LiveConnectConfig(
            response_modalities=[Modality.AUDIO],
            input_audio_transcription=AudioTranscriptionConfig(),
            output_audio_transcription=AudioTranscriptionConfig(),
            speech_config=SpeechConfig(language_code="en-US"),
            proactivity=ProactivityConfig(enabled=True),
            system_instruction=Content(parts=[Part(text=(
                "You are a helpful assistant talking to a user through a robot named Reachy. "
                "The user can see and talk to Reachy, and Reachy can talk back and listen. "
                "Keep your responses short and concise, ideally under 20 seconds of audio. "
                "If you need to say more, break it up into multiple responses. "
                "Only respond when asked a direct question."
            ))]),
            tools=[
                Tool(
                    function_declarations=[
                        FunctionDeclaration(
                            name="end_conversation",
                            description="End the conversation with Gemini Live."
                        )
                    ]
                )
            ],
        )
        async with client.aio.live.connect(model=MODEL, config=config) as session:
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
                await receive_loop(session, speaker_queue, interrupted_event)
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
