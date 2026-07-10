import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import time


def get_device_info(device_index):
    """Get supported channels and sample rate for a device."""
    info = sd.query_devices(device_index)
    channels = min(2, int(info['max_input_channels']))
    samplerate = int(info['default_samplerate'])
    return channels, samplerate


def calibrate(input_device, duration=10):
    """
    Listen to game audio for `duration` seconds and identify
    the dominant engine harmonic frequencies using FFT analysis.
    Returns a list of suggested notch frequencies.
    """
    channels, samplerate = get_device_info(input_device)
    print(f"Device channels: {channels}, sample rate: {samplerate}")
    print(f"Listening for {duration} seconds — rev the engine...")

    frames = []

    def callback(indata, f, t, status):
        frames.append(indata.copy())

    with sd.InputStream(
        device=input_device,
        samplerate=samplerate,
        channels=channels,
        dtype='float32',
        callback=callback
    ):
        time.sleep(duration)

    audio = np.concatenate(frames)
    # Use first channel regardless of mono/stereo
    audio_mono = audio[:, 0]

    fft_mag = np.abs(np.fft.rfft(audio_mono))
    freqs   = np.fft.rfftfreq(len(audio_mono), 1/samplerate)

    fft_smooth = uniform_filter1d(fft_mag, size=20)

    mask = (freqs > 80) & (freqs < 1000)
    if mask.sum() == 0:
        print("No frequencies detected in range")
        return [200, 320, 400, 600, 800]

    peaks, _ = find_peaks(
        fft_smooth[mask],
        height=np.percentile(fft_smooth[mask], 75),
        distance=30,
        prominence=fft_smooth[mask].max() * 0.1
    )

    peak_freqs = freqs[mask][peaks]
    peak_freqs = sorted([int(round(f/10)*10) for f in peak_freqs])

    if not peak_freqs:
        print("No clear harmonics detected — using defaults")
        return [200, 320, 400, 600, 800]

    print(f"\nDetected engine harmonics: {peak_freqs} Hz")
    print(f"Suggested: --notch-freqs {' '.join(str(f) for f in peak_freqs)}")

    return peak_freqs
