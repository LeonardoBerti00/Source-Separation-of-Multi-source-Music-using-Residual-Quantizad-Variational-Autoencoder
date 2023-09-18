import av
import numpy as np


def compute_output_dim_conv(input_dim, kernel_size, padding, dilation, stride):
    return ((input_dim + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)

def compute_output_dim_convtranspose(input_dim, kernel_size, padding, dilation, stride):
    return (input_dim - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1


def load_audio(file,  offset, duration, sr=None, resample=False, approx=False, time_base='samples', check_duration=True):
    if time_base == 'sec':
        offset = offset * sr
        duration = duration * sr
    # Loads at target sr, stereo channels, seeks from offset, and stops after duration
    container = av.open(file)
    audio = container.streams.get(audio=0)[0] # Only first audio stream
    audio_duration = audio.duration * float(audio.time_base)
    if approx:
        if offset + duration > audio_duration*sr:
            # Move back one window. Cap at audio_duration
            offset = np.min(audio_duration*sr - duration, offset - duration)
    else:
        if check_duration:
            assert offset + duration <= audio_duration*sr, f'End {offset + duration} beyond duration {audio_duration*sr}'
    if resample:
        resampler = av.AudioResampler(format='fltp',layout='stereo', rate=sr)
    else:
        assert sr == audio.sample_rate
    offset = int(offset / sr / float(audio.time_base)) #int(offset / float(audio.time_base)) # Use units of time_base for seeking
    sig = np.zeros((2, duration), dtype=np.float32)
    container.seek(offset, stream=audio)
    total_read = 0
    for frame in container.decode(audio=0): # Only first audio stream
        if resample:
            frame.pts = None
            frame = resampler.resample(frame)
        frame = frame.to_ndarray(format='fltp') # Convert to floats and not int16
        read = frame.shape[-1]
        if total_read + read > duration:
            read = duration - total_read
        sig[:, total_read:total_read + read] = frame[:, :read]
        total_read += read
        if total_read == duration:
            break
    assert total_read <= duration, f'Expected {duration} frames, got {total_read}'
    return sig, sr

