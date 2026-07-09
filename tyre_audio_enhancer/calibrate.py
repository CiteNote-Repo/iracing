import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
import time

def calibrate(input_device, duration=10):
    """
    Listen to game audio for `duration` seconds and identify
    the dominant engine harmonic frequencies using FFT analysis.
    Returns a list of suggested notch frequencies.
    """
    SAMPLE_RATE = 48000
    print(f"Listening for {duration} seconds — rev the engine...")

    frames = []

    def callback(indata, f, t, status):
        frames.append(indata.copy())

    with sd.InputStream(
        device=input_device,
        samplerate=SAMPLE_RATE,
        channels=2,
        callback=callback
    ):
        time.sleep(duration)

    # Combine all frames and take left channel
    audio = np.concatenate(frames)[:, 0]

    # Compute FFT magnitude spectrum
    fft_mag = np.abs(np.fft.rfft(audio))
    freqs   = np.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)

    # Smooth the spectrum
    from scipy.ndimage import uniform_filter1d
    fft_smooth = uniform_filter1d(fft_mag, size=20)

    # Find peaks in the 80-1000Hz engine range
    mask = (freqs > 80) & (freqs < 1000)
    peaks, props = find_peaks(
        fft_smooth[mask],
        height=np.percentile(fft_smooth[mask], 75),
        distance=30,
        prominence=fft_smooth[mask].max() * 0.1
    )

    peak_freqs = freqs[mask][peaks]
    peak_freqs = sorted([int(round(f/10)*10) for f in peak_freqs])

    print(f"\nDetected engine harmonics: {peak_freqs} Hz")
    print(f"Suggested: --notch-freqs {' '.join(str(f) for f in peak_freqs)}")

    return peak_freqs
