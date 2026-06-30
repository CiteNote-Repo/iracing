import math
import threading
from typing import Optional

try:
    import numpy as np
    import sounddevice as sd
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class GripToneSynth:
    """
    Continuous stereo sine-wave synthesiser.
    LEFT channel = front axle utilisation.
    RIGHT channel = rear axle utilisation.
    Frequency glides smoothly to avoid zipper noise.
    Phase is tracked in radians so it stays continuous across blocks.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 512,
        volume: float = 0.15,
    ):
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._volume = volume

        # Freq state — shared with telemetry thread via lock
        self._front_freq: float = 110.0
        self._rear_freq: float = 110.0
        self._front_freq_target: float = 110.0
        self._rear_freq_target: float = 110.0
        self._muted: bool = True
        self._lock = threading.Lock()

        # Phase state — only ever touched from the audio callback thread
        self._front_phase: float = 0.0  # radians
        self._rear_phase: float = 0.0

        self._stream = None

    @staticmethod
    def available() -> bool:
        return _AVAILABLE

    def set_utilization(self, front_pct: float, rear_pct: float, active: bool) -> None:
        """Called from telemetry thread to update target frequencies."""
        with self._lock:
            self._muted = not active
            if active:
                self._front_freq_target = self._util_to_freq(front_pct)
                self._rear_freq_target = self._util_to_freq(rear_pct)

    @staticmethod
    def _util_to_freq(util_pct: float) -> float:
        """
        Map 0-130% → 110-880 Hz with an exponential-ish curve:
          0-50%:   110-165 Hz  (quiet background presence)
          50-90%:  165-440 Hz  (approaching limit)
          90-100%: 440-660 Hz  (critical zone, steep)
          100%+:   660-880 Hz  (over-limit, capped)
        """
        u = max(0.0, min(util_pct, 130.0))
        if u < 50.0:
            return 110.0 + (u / 50.0) * 55.0
        elif u < 90.0:
            return 165.0 + ((u - 50.0) / 40.0) * 275.0
        elif u < 100.0:
            return 440.0 + ((u - 90.0) / 10.0) * 220.0
        else:
            return 660.0 + min((u - 100.0) / 30.0, 1.0) * 220.0

    def _audio_callback(self, outdata, frames, time_info, status):
        # Acquire lock only for scalar glide + copy — not during numpy ops
        with self._lock:
            if self._muted:
                outdata[:] = 0.0
                # Keep frequencies gliding toward targets even while muted so
                # resumption at speed is already converged
                glide = 0.10
                self._front_freq += (self._front_freq_target - self._front_freq) * glide
                self._rear_freq += (self._rear_freq_target - self._rear_freq) * glide
                return

            glide = 0.15
            self._front_freq += (self._front_freq_target - self._front_freq) * glide
            self._rear_freq += (self._rear_freq_target - self._rear_freq) * glide
            ff = self._front_freq
            rf = self._rear_freq
            vol = self._volume

        sr = self._sample_rate
        two_pi = 2.0 * math.pi

        # Phase-continuous block synthesis
        # n = sample indices within this block
        n = np.arange(frames, dtype=np.float64)

        front_wave = (vol * np.sin(two_pi * ff / sr * n + self._front_phase)).astype(np.float32)
        self._front_phase = math.fmod(
            self._front_phase + two_pi * ff * frames / sr, two_pi
        )

        rear_wave = (vol * np.sin(two_pi * rf / sr * n + self._rear_phase)).astype(np.float32)
        self._rear_phase = math.fmod(
            self._rear_phase + two_pi * rf * frames / sr, two_pi
        )

        # LEFT = front, RIGHT = rear
        outdata[:, 0] = front_wave
        outdata[:, 1] = rear_wave

    def start(self) -> None:
        if not _AVAILABLE:
            raise RuntimeError(
                "sounddevice / numpy not installed.\n"
                "Run: pip install sounddevice numpy"
            )
        self._stream = sd.OutputStream(
            samplerate=self._sample_rate,
            channels=2,
            callback=self._audio_callback,
            blocksize=self._blocksize,
            dtype="float32",
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
