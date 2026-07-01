import math
import threading

try:
    import numpy as np
    import sounddevice as sd
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_PULSE_FREQ = 80.0   # Hz — rear slip alert tone
_PULSE_VOL  = 0.25   # independent of main volume
_PULSE_DUR  = 0.08   # seconds
_SLIP_ON    = 0.03   # rising threshold
_SLIP_OFF   = 0.02   # hysteresis reset


class GripToneSynth:
    """
    Continuous stereo synthesiser driven by three grip metrics.

    Both channels carry the same pitch (total utilisation).
    Steering efficiency adds a second harmonic, making the tone subtly
    rougher when the front is scrubbing.
    Rear slip fires a single 80 Hz pulse in the right ear on the rising edge.
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

        # Shared state — written by telemetry thread, read by audio callback
        self._freq: float = 110.0
        self._freq_target: float = 110.0
        self._harmonic_mix: float = 0.0      # 0.0, 0.05, or 0.10
        self._muted: bool = True
        self._slip_above: bool = False        # hysteresis state
        self._pulse_samples_remaining: int = 0
        self._lock = threading.Lock()

        # Phase state — audio callback thread only, never touched under lock
        self._phase: float = 0.0
        self._harmonic_phase: float = 0.0
        self._pulse_phase: float = 0.0

        self._stream = None

    @staticmethod
    def available() -> bool:
        return _AVAILABLE

    def set_state(
        self,
        total_util: float,
        scrub_proximity_pct: float,
        rear_slip_raw: float,
        active: bool,
    ) -> None:
        """Called from telemetry thread ~60 Hz."""
        with self._lock:
            self._muted = not active
            if not active:
                return

            self._freq_target = self._util_to_freq(total_util)

            # scrub_proximity_pct == 0 means no data (gates not met) — stay pure.
            # >80 = front efficient → clean tone.
            # 60-80 = approaching scrub → subtle harmonic.
            # <60 = past scrub peak → clearly rougher tone.
            if scrub_proximity_pct <= 0 or scrub_proximity_pct > 80.0:
                self._harmonic_mix = 0.0
            elif scrub_proximity_pct >= 60.0:
                self._harmonic_mix = 0.05
            else:
                self._harmonic_mix = 0.15

            # Slip pulse: fire once on rising edge, reset on hysteresis fall
            if rear_slip_raw > _SLIP_ON and not self._slip_above:
                self._slip_above = True
                self._pulse_samples_remaining = int(_PULSE_DUR * self._sample_rate)
            elif rear_slip_raw < _SLIP_OFF:
                self._slip_above = False

    @staticmethod
    def _util_to_freq(util_pct: float) -> float:
        """
        Map 0-130% → 110-880 Hz:
          0-50%:   110-165 Hz
          50-90%:  165-440 Hz
          90-100%: 440-660 Hz
          100%+:   660-880 Hz
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
        with self._lock:
            if self._muted:
                outdata[:] = 0.0
                # Keep gliding toward target so resumption is already converged
                self._freq += (self._freq_target - self._freq) * 0.25
                return

            self._freq += (self._freq_target - self._freq) * 0.40
            freq          = self._freq
            vol           = self._volume
            harmonic_mix  = self._harmonic_mix
            pulse_samples = min(self._pulse_samples_remaining, frames)
            self._pulse_samples_remaining -= pulse_samples

        sr     = self._sample_rate
        two_pi = 2.0 * math.pi
        n      = np.arange(frames, dtype=np.float64)

        # Fundamental + optional second harmonic, normalised to constant peak
        fundamental = np.sin(two_pi * freq / sr * n + self._phase)
        harmonic2   = np.sin(two_pi * 2.0 * freq / sr * n + self._harmonic_phase)
        wave = vol * (fundamental + harmonic_mix * harmonic2) / (1.0 + harmonic_mix)

        # Advance phases — harmonic_phase advances even when mix=0 so it is
        # already in the right place when efficiency drops and it kicks in
        self._phase = math.fmod(
            self._phase + two_pi * freq * frames / sr, two_pi
        )
        self._harmonic_phase = math.fmod(
            self._harmonic_phase + two_pi * 2.0 * freq * frames / sr, two_pi
        )

        wave = wave.astype(np.float32)
        outdata[:, 0] = wave
        outdata[:, 1] = wave.copy()

        # Right-ear slip pulse — additive, right channel only
        if pulse_samples > 0:
            p     = np.arange(pulse_samples, dtype=np.float64)
            pulse = (_PULSE_VOL * np.sin(
                two_pi * _PULSE_FREQ / sr * p + self._pulse_phase
            )).astype(np.float32)
            self._pulse_phase = math.fmod(
                self._pulse_phase + two_pi * _PULSE_FREQ * pulse_samples / sr, two_pi
            )
            outdata[:pulse_samples, 1] += pulse

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
