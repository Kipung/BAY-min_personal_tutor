from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress

from google import genai
from reachy_mini import ReachyMini

from reachy_communication.audio_adapters import (
    PCMFramer,
    pcm16_mono_24k_to_reachy_float32_stereo_44k1,
    reachy_float32_stereo_to_pcm16_mono_16k,
)
from reachy_communication.gemini_live import (
    extract_audio_chunks,
    extract_input_transcript,
    extract_output_transcript,
    is_interrupted,
)

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
LIVE_CONFIG = {
    "response_modalities": ["AUDIO", "TEXT"],
    "input_audio_transcription": {},
    "output_audio_transcription": {},
}
SPEAKER_QUEUE_MAX = 60
MIC_QUEUE_MAX = 6
PLAY_CHUNK_SECONDS = 0.02
REACHY_SR = 44100


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
    while True:
        audio = mini.media.get_audio_sample()
        if audio is None:
            await asyncio.sleep(0.002)
            continue

        pcm16_16k = reachy_float32_stereo_to_pcm16_mono_16k(audio)
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
        await session.send_realtime_input(audio={"data": frame, "mime_type": "audio/pcm"})


async def receive_loop(
    session,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
) -> None:
    """
    Receive Gemini Live responses, extract audio and text, and push audio to speaker_queue.
    """
    async for response in session.receive():
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
            #placeholder for uploading to firestore
            print(f"ASSISTANT_SPOKEN: {spoken_tx}")

        audio_chunks = extract_audio_chunks(response)
        if audio_chunks:
            interrupted_event.clear()
            for chunk in audio_chunks:
                drop_oldest_put_nowait(speaker_queue, chunk)


async def play_speaker_loop(
    mini: ReachyMini,
    speaker_queue: asyncio.Queue,
    interrupted_event: asyncio.Event,
) -> None:
    """
    Read PCM16 24kHz mono audio chunks from speaker_queue, convert to Reachy format, and play.
    """
    slice_n = int(REACHY_SR * PLAY_CHUNK_SECONDS)
    while True:
        audio_24k_pcm16 = await speaker_queue.get()

        if interrupted_event.is_set():
            continue

        stereo_44k1 = pcm16_mono_24k_to_reachy_float32_stereo_44k1(audio_24k_pcm16)

        for start in range(0, stereo_44k1.shape[0], slice_n):
            if interrupted_event.is_set():
                break
            mini.media.push_audio_sample(stereo_44k1[start : start + slice_n])
            await asyncio.sleep(0)


async def run() -> None:
    with ReachyMini() as mini:
        mini.media.start_recording()
        mini.media.start_playing()

        client = genai.Client()
        async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:
            mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_MAX)
            speaker_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=SPEAKER_QUEUE_MAX)
            interrupted_event = asyncio.Event()

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
