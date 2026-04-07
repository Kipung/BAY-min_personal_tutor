from __future__ import annotations

import asyncio

import cv2
import numpy as np
from reachy_mini import ReachyMini

VISION_CAPTURE_INTERVAL_S = 0.5  # internal grab rate (~2 fps)
VISION_JPEG_QUALITY = 95         # higher = better quality for LLM interpretation


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
            if frame is not None and not self._is_white(frame):
                jpeg_bytes = self._encode_jpeg(frame)
                if jpeg_bytes:
                    async with self._lock:
                        self._latest_frame_bytes = jpeg_bytes
            await asyncio.sleep(VISION_CAPTURE_INTERVAL_S)

    @staticmethod
    def _is_white(frame: np.ndarray, threshold: float = 250.0) -> bool:
        return float(np.mean(frame)) >= threshold

    def _encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
        success, buf = cv2.imencode(
            ".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, VISION_JPEG_QUALITY]
        )
        if not success:
            return None
        return buf.tobytes()

    async def get_latest_frame_bytes(self) -> bytes | None:
        async with self._lock:
            return self._latest_frame_bytes

