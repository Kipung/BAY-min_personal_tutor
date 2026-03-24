from __future__ import annotations

import asyncio

import cv2
import numpy as np
from google import genai
from reachy_mini import ReachyMini

VISION_CAPTURE_INTERVAL_S = 0.1  # internal grab rate (~10 fps)
VISION_SEND_INTERVAL_S = 2.0     # how often a frame is pushed to Gemini Live
VISION_JPEG_QUALITY = 80         # lower = smaller payload


class ReachyVision:
    def __init__(self, mini: ReachyMini) -> None:
        self._media = mini.media
        self._latest_frame_bytes: bytes | None = None
        self._lock = asyncio.Lock()

    async def capture_loop(self) -> None:
        """Continuously grabs frames from Reachy's camera and stores the latest as JPEG bytes."""
        loop = asyncio.get_running_loop()
        while True:
            # get_frame() is synchronous/blocking — run it off the event loop thread
            frame: np.ndarray | None = await loop.run_in_executor(
                None, self._media.get_frame
            )
            if frame is not None:
                jpeg_bytes = self._encode_jpeg(frame)
                if jpeg_bytes:
                    async with self._lock:
                        self._latest_frame_bytes = jpeg_bytes
            await asyncio.sleep(VISION_CAPTURE_INTERVAL_S)

    def _encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        success, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, VISION_JPEG_QUALITY]
        )
        if not success:
            return None
        return buf.tobytes()

    async def get_latest_frame_bytes(self) -> bytes | None:
        async with self._lock:
            return self._latest_frame_bytes


async def send_vision_loop(session, vision: ReachyVision) -> None:
    """Periodically sends the latest camera frame to the active Gemini Live session."""
    while True:
        await asyncio.sleep(VISION_SEND_INTERVAL_S)
        frame_bytes = await vision.get_latest_frame_bytes()
        if frame_bytes is not None:
            await session.send_realtime_input(
                video=genai.types.Blob(data=frame_bytes, mime_type="image/jpeg")
            )
