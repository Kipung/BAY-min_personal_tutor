import time
import torch
import numpy as np
from math import gcd
import soundfile as sf

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
from silero_vad import load_silero_vad, VADIterator
from scipy.signal import resample_poly


TARGET_SR = 16000
FRAME_SAMPLES = 512  # required by VADIterator at 16k


def to_mono_float32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:  # (n, channels)
        x = x.mean(axis=1)

    # Common formats: int16 PCM or float
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32)

    # Defensive normalize if it looks like unnormalized ints in float container
    m = np.max(np.abs(x)) if x.size else 0.0
    if m > 1.5:
        x = x / m

    return x


def resample_to_target(x: np.ndarray, sr_in: int, sr_out: int = TARGET_SR) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down).astype(np.float32)


def main():
    model = load_silero_vad()

    # rolling buffer of float32 audio @ 16k
    buf = np.zeros((0,), dtype=np.float32)
    out = None

    vad = VADIterator(
        model,
        sampling_rate=TARGET_SR,
        threshold=0.5,
        min_silence_duration_ms=300
    )

    with ReachyMini() as mini:
        mini.media.start_recording()

        # Best-effort: get the input samplerate if your SDK exposes it
        try:
            sr_in = int(mini.media.get_input_audio_samplerate())
        except Exception:
            sr_in = TARGET_SR  # assume already 16k if not available

        print(f"Mic SR: {sr_in} Hz -> feeding VAD {TARGET_SR} Hz frames of {FRAME_SAMPLES} samples")

        try:
            i = 0
            while True:
                chunk = mini.media.get_audio_sample()
                if chunk is None:
                    time.sleep(0.005)
                    continue
                if out is None:
                    out = chunk
                else:
                    out = np.concatenate([out, chunk])

                x = to_mono_float32(chunk)
                x = resample_to_target(x, sr_in, TARGET_SR)

                # append to buffer
                if x.size:
                    buf = np.concatenate([buf, x])

                # feed fixed 512-sample frames
                while buf.size >= FRAME_SAMPLES:
                    frame = buf[:FRAME_SAMPLES]
                    buf = buf[FRAME_SAMPLES:]

                    event = vad(frame)
                    if event is not None:
                        if 'start' in event:
                            mini.goto_target(head=create_head_pose(0,0,0), antennas=(0, 0), duration=0.1)
                        else:
                            mini.goto_target(head=create_head_pose(0,0,0), antennas=(-3.14/6, 3.14/6), duration=0.1)

                time.sleep(0.002)  # tiny yield to keep CPU sane
        except KeyboardInterrupt:
            pass
    #save out to audio file
    sf.write("vad_output.wav", out, sr_in)

if __name__ == "__main__":
    main()
