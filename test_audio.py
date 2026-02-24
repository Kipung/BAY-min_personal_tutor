from reachy_mini import ReachyMini

with ReachyMini() as mini:
    print(f"input rate: {mini.media.get_input_audio_samplerate()} channels: {mini.media.get_input_channels()}")
    print(f"output rate: {mini.media.get_output_audio_samplerate()} channels: {mini.media.get_output_channels()}")