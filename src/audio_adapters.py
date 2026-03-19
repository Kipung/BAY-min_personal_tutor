import numpy as np
from scipy.signal import resample_poly

UPLINK_SR = 16000
FRAME_MS = 20
UPLINK_SAMPLES_PER_FRAME = int(UPLINK_SR * FRAME_MS / 1000)  # 320
UPLINK_BYTES_PER_FRAME = UPLINK_SAMPLES_PER_FRAME * 2         # int16 mono

def resample_to_16k_mono(audio: np.ndarray, input_sr: int, input_ch: int) -> bytes:
    """
    audio: float32 array shape (n, 2) in [-1, 1], input_sr
    returns: PCM16 little-endian mono bytes at 16kHz
    """
    if audio.ndim != 2 or audio.shape[1] != input_ch:

        raise ValueError(f"Expected stereo float32 (n,{input_ch}), got {audio.shape}, dtype={audio.dtype}")

    # stereo -> mono
    mono = audio.mean(axis=1).astype(np.float32)

    if input_sr != UPLINK_SR:
        # resample input -> 16000
        mono_16k = resample_poly(mono, up=16000, down=input_sr).astype(np.float32)
    else:
        mono_16k = mono

    # float -> pcm16
    mono_16k = np.clip(mono_16k, -1.0, 1.0)
    pcm16 = (mono_16k * 32767.0).astype(np.int16)
    return pcm16.tobytes()

class PCMFramer:
    """
    Accumulate bytes and yield fixed-size frames. Keeps leftover bytes for the next frame.
    Frame size is determined by UPLINK_BYTES_PER_FRAME (e.g. 640 bytes for 20ms of 16kHz mono PCM16).
    """
    def __init__(self):
        self.buf = bytearray()

    def push(self, chunk: bytes):
        self.buf.extend(chunk)

    def pop_frames(self):
        while len(self.buf) >= UPLINK_BYTES_PER_FRAME:
            frame = bytes(self.buf[:UPLINK_BYTES_PER_FRAME])
            del self.buf[:UPLINK_BYTES_PER_FRAME]
            yield frame

def resample_from_24kHz(audio_bytes: bytes, output_sr: int) -> np.ndarray:
    """
    audio_bytes: PCM16 mono @ 24kHz
    returns: float32 stereo @ output_sr in [-1,1], shape (n,2)
    """
    x_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    x = x_i16.astype(np.float32) / 32768.0

    # resample 24000 -> output_sr
    y = resample_poly(x, up=output_sr, down=24000).astype(np.float32)
    y = np.clip(y, -1.0, 1.0)

    # mono -> stereo
    stereo = np.column_stack([y, y]).astype(np.float32)
    return stereo
