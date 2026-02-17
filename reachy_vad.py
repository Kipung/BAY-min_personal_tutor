from reachy_mini import ReachyMini
from scipy.signal import resample
import time

with ReachyMini(media_backend="default") as mini:
    input(
        "Press Enter to start recording and playing audio. The audio devices will be busy until the end of the program."
    )
    # Initialization - After this point, both audio devices (input/output) will be seen as busy by other applications!
    mini.media.start_recording()
    mini.media.start_playing()

    # Record: first poll may return None if callback has not filled the buffer yet.
    samples = None
    deadline = time.time() + 3.0
    while time.time() < deadline:
        samples = mini.media.get_audio_sample()
        if samples is not None and len(samples) > 0:
            break
        time.sleep(0.05)

    if samples is None or len(samples) == 0:
        raise RuntimeError(
            "No audio samples were captured. Check your input device and try again."
        )

    # Resample (if needed)
    input_sr = mini.media.get_input_audio_samplerate()
    output_sr = mini.media.get_output_audio_samplerate()
    target_len = int(round(output_sr * len(samples) / input_sr))
    if target_len <= 0:
        raise RuntimeError("Computed invalid resample length.")
    print(target_len)
    samples = resample(samples, target_len)

    # Play
    mini.media.push_audio_sample(samples)
    time.sleep(len(samples) / output_sr)

    # Get Direction of Arrival
    # 0 radians is left, π/2 radians is front/back, π radians is right.
    doa_result = mini.media.get_DoA()
    if doa_result is None:
        print("DoA unavailable: ReSpeaker / Reachy Mini Audio device not detected.")
    else:
        doa, is_speech_detected = doa_result
        print(doa, is_speech_detected)

    # Release audio devices (input/output)
    mini.media.stop_recording()
    mini.media.stop_playing()
