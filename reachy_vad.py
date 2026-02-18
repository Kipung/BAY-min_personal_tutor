from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np
from scipy.signal import resample
import time


def doa_to_yaw(doa: float) -> float:
    """Convert ReSpeaker DoA (0=left, pi/2=front, pi=right) to robot yaw (0=front)."""
    return float(doa - (np.pi / 2.0))


with ReachyMini() as mini:
    # Initialization - After this point, both audio devices (input/output) will be seen as busy by other applications!
    mini.media.start_recording()
    mini.goto_target(
        body_yaw=0.0,
        head=create_head_pose(pitch=-0.5, yaw=0.0),
        duration=1.0,
    )
    # mini.media.start_playing()
    while True:
        if input(
            "Press Enter to start recording and playing audio. The audio devices will be busy until the end of the program."
        ) == "exit":
            break

        # Record for the full duration by accumulating chunks.
        chunks = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            chunk = mini.media.get_audio_sample()
            
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
                break #remove this line to record the full 5 seconds, or keep it to only record the first chunk of audio.
            time.sleep(0.05)

        if len(chunks) == 0:
            raise RuntimeError(
                "No audio samples were captured. Check your input device and try again."
            )
        samples = np.concatenate(chunks)

        # Resample (if needed)
        input_sr = mini.media.get_input_audio_samplerate()
        output_sr = mini.media.get_output_audio_samplerate()
        target_len = int(round(output_sr * len(samples) / input_sr))
        if target_len <= 0:
            raise RuntimeError("Computed invalid resample length.")
        # print(target_len)
        samples = resample(samples, target_len)

        # print(f"Recorded {len(samples)} samples at {input_sr} Hz, resampled to {output_sr} Hz.")
        # print(f"playing {len(samples) / output_sr} seconds of audio...")
        # Play
        # mini.media.push_audio_sample(samples)
        # time.sleep(len(samples) / output_sr)

        # Get Direction of Arrival
        # 0 radians is left, π/2 radians is front/back, π radians is right.
        doa_result = mini.media.get_DoA()
        if doa_result is None:
            print("DoA unavailable: ReSpeaker / Reachy Mini Audio device not detected.")
        else:
            doa, is_speech_detected = doa_result
            yaw = doa_to_yaw(doa)
            print(f"doa={doa:.3f} yaw={yaw:.3f} speech={is_speech_detected}")
            # if is_speech_detected:
            if True:
                mini.goto_target(
                    body_yaw=yaw,
                    # Keep head aligned with the body frame while base turns to DoA.
                    head=create_head_pose(yaw=0.0),
                    duration=1.0,
                )

    # Release audio devices (input/output)
    mini.media.stop_recording()
    # mini.media.stop_playing()
