from __future__ import annotations

import asyncio
import logging

import cv2
import numpy as np
from reachy_mini import ReachyMini
from reachy_mini.media.camera_constants import CameraResolution

logger = logging.getLogger(__name__)

VISION_CAPTURE_INTERVAL_S = 0.5  # internal grab rate (~2 fps)
VISION_JPEG_QUALITY = 98         # high quality for reading text on paper

# --- Face detection ---
FACE_DETECT_SCALE = (320, 240)   # downscale for fast detection
FACE_MIN_SIZE = (40, 40)

# Preferred resolutions, ordered by preference — highest res first.
# The camera reports: 1920x1080@60, 3840x2592@30, 3840x2160@30, 3264x2448@30
PREFERRED_RESOLUTIONS = [
    CameraResolution.R3840x2592at30fps,
    CameraResolution.R3840x2160at30fps,
    CameraResolution.R3264x2448at30fps,
    CameraResolution.R2304x1296at30fps,
    CameraResolution.R1920x1080at30fps,
    CameraResolution.R1920x1080at60fps,
    CameraResolution.R1280x720at30fps,
]


class ReachyVision:
    def __init__(self, mini: ReachyMini) -> None:
        self._media = mini.media
        self._latest_frame_raw: np.ndarray | None = None
        self._lock = asyncio.Lock()

        # Face detection via OpenCV Haar cascade (lightweight, no extra deps)
        self._face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._resolution_configured = False

    def _try_set_higher_resolution(self) -> None:
        """Attempt to set the camera to a higher resolution than the default 720p."""
        cam = self._media.camera
        if cam is None or cam.camera_specs is None:
            return

        cap = getattr(cam, "cap", None)

        if cap is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[vision] Camera default resolution: {actual_w}x{actual_h}")

        available = cam.camera_specs.available_resolutions
        print(f"[vision] Available resolutions: {[(r.value[0], r.value[1], r.value[2]) for r in available]}")

        for res in PREFERRED_RESOLUTIONS:
            if res in available:
                try:
                    cam.set_resolution(res)
                    if cap is not None:
                        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"[vision] Requested {res.value[0]}x{res.value[1]}, got {actual_w}x{actual_h}")
                        if actual_w == res.value[0] and actual_h == res.value[1]:
                            break
                        else:
                            print(f"[vision] Resolution change did not take effect, trying next...")
                            continue
                    break
                except Exception as e:
                    print(f"[vision] Failed to set resolution {res}: {e}")

        if cap is not None:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[vision] Final camera resolution: {actual_w}x{actual_h}")
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    async def capture_loop(self) -> None:
        """Continuously grabs frames and stores the latest raw frame (no encoding)."""
        loop = asyncio.get_running_loop()
        logged_size = False
        while True:
            if not self._resolution_configured:
                self._try_set_higher_resolution()
                self._resolution_configured = True

            frame: np.ndarray | None = await loop.run_in_executor(
                None, self._media.get_frame
            )
            if frame is not None and not self._is_white(frame):
                if not logged_size:
                    print(f"[vision] Actual frame from get_frame(): {frame.shape[1]}x{frame.shape[0]}")
                    logged_size = True
                async with self._lock:
                    self._latest_frame_raw = frame
            await asyncio.sleep(VISION_CAPTURE_INTERVAL_S)

    async def get_latest_frame_bytes(self) -> bytes | None:
        """Encode the latest raw frame to JPEG on demand (only when capture_image is called)."""
        async with self._lock:
            frame = self._latest_frame_raw
        if frame is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_jpeg, frame)

    async def get_face_center(self) -> tuple[int, int] | None:
        """
        Return (u, v) pixel coordinates of the largest face in the latest frame,
        in the original frame resolution. Returns None if no face detected.
        """
        async with self._lock:
            frame = self._latest_frame_raw
        if frame is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._detect_face_center, frame)

    def _detect_face_center(self, frame: np.ndarray) -> tuple[int, int] | None:
        """Detect the largest face and return its center in original frame coordinates."""
        h_orig, w_orig = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, FACE_DETECT_SCALE)

        faces = self._face_cascade.detectMultiScale(
            small, scaleFactor=1.1, minNeighbors=4, minSize=FACE_MIN_SIZE,
        )
        if len(faces) == 0:
            return None

        areas = [w * h for (_, _, w, h) in faces]
        idx = int(np.argmax(areas))
        x, y, w, h = faces[idx]

        scale_x = w_orig / FACE_DETECT_SCALE[0]
        scale_y = h_orig / FACE_DETECT_SCALE[1]
        cx = int((x + w / 2) * scale_x)
        cy = int((y + h / 2) * scale_y)
        return (cx, cy)

    @staticmethod
    def _is_white(frame: np.ndarray, threshold: float = 250.0) -> bool:
        return float(np.mean(frame)) >= threshold

    @staticmethod
    def _encode_jpeg(frame: np.ndarray) -> bytes | None:
        success, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, VISION_JPEG_QUALITY]
        )
        if not success:
            return None
        return buf.tobytes()
