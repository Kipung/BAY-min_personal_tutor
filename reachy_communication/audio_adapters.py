import numpy as np
from scipy.signal import resample_poly

IN_SR = 44100
IN_CH = 2
UPLINK_SR = 16000
FRAME_MS = 20
UPLINK_SAMPLES_PER_FRAME = int(UPLINK_SR * FRAME_MS / 1000)  # 320
UPLINK_BYTES_PER_FRAME = UPLINK_SAMPLES_PER_FRAME * 2         # int16 mono

def reachy_float32_stereo_to_pcm16_mono_16k(audio: np.ndarray) -> bytes:
    """
    audio: float32 array shape (n, 2) in [-1, 1], 44.1kHz
    returns: PCM16 little-endian mono bytes at 16kHz
    """
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"Expected stereo float32 (n,2), got {audio.shape}, dtype={audio.dtype}")

    # stereo -> mono
    mono = audio.mean(axis=1).astype(np.float32)

    # resample 44100 -> 16000 (ratio 441/160)
    mono_16k = resample_poly(mono, up=160, down=441).astype(np.float32)

    # float -> pcm16
    mono_16k = np.clip(mono_16k, -1.0, 1.0)
    pcm16 = (mono_16k * 32767.0).astype(np.int16)
    return pcm16.tobytes()

class PCMFramer:
    """
    Accumulate bytes and yield fixed-size frames.
    Gemini Live API reccomends 20-40ms frames
    """
    def __init__(self, frame_bytes: int):
        self.frame_bytes = frame_bytes
        self.buf = bytearray()

    def push(self, chunk: bytes):
        self.buf.extend(chunk)

    def pop_frames(self):
        while len(self.buf) >= self.frame_bytes:
            frame = bytes(self.buf[:self.frame_bytes])
            del self.buf[:self.frame_bytes]
            yield frame

def pcm16_mono_24k_to_reachy_float32_stereo_44k1(audio_bytes: bytes) -> np.ndarray:
    """
    audio_bytes: PCM16 mono @ 24kHz
    returns: float32 stereo @ 44.1kHz in [-1,1], shape (n,2)
    """
    x_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    x = x_i16.astype(np.float32) / 32768.0

    # resample 24000 -> 44100 (ratio 147/80)
    y = resample_poly(x, up=147, down=80).astype(np.float32)
    y = np.clip(y, -1.0, 1.0)

    # mono -> stereo
    stereo = np.column_stack([y, y]).astype(np.float32)
    return stereo