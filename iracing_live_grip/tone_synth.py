import math
import threading

try:
    import numpy as np
    import sounddevice as sd
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SLIP_ON  = 15.0   # yaw deviation % — oversteer burst trigger
_SLIP_OFF = 8.0


def _util_to_freq(util: float) -> float:
    """
    Non-linear pitch mapping — dedicates most pitch space to
    the 70-100% zone where limit sensitivity matters most.

    0%  → 200Hz  (barely audible presence)
    50% → 280Hz  (building, still background)
    70% → 350Hz  (approaching limit starts here)
    90% → 620Hz  (rapid climb through critical zone)
    100% → 800Hz (at limit)
    >100%: stays at 800Hz — timbre change signals over-limit
            (add 15% of a 2× harmonic when util > 100)
    """
    util = max(0.0, util)
    if util <= 50.0:
        return 200.0 + util * 1.6           # 200→280Hz, slow
    elif util <= 70.0:
        return 280.0 + (util - 50.0) * 3.5  # 280→350Hz, gentle
    elif util <= 90.0:
        return 350.0 + (util - 70.0) * 13.5 # 350→620Hz, rapid
    elif util <= 100.0:
        return 620.0 + (util - 90.0) * 18.0 # 620→800Hz, steep
    else:
        return 800.0  # capped — harmonic added below


class GripToneSynth:

    def __init__(self, sample_rate: int = 44100, blocksize: int = 512, volume: float = 0.08):
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._volume = volume

        self._muted = True
        self._total_util = 0.0
        self._slip_above = False
        self._burst_pending = False
        self._lock = threading.Lock()

        # Audio callback state only — no lock needed
        self._phase = 0.0
        self._current_freq = 200.0
        self._burst_samples = None  # pre-rendered burst array
        self._burst_pos = 0

        self._stream = None

    @staticmethod
    def available() -> bool:
        return _AVAILABLE

    def set_state(
        self,
        total_util: float,
        scrub_proximity_pct: float,
        yaw_deviation_pct: float,
        active: bool,
    ) -> None:
        with self._lock:
            self._muted = not active
            if not active:
                return
            self._total_util = total_util
            if yaw_deviation_pct > _SLIP_ON and not self._slip_above:
                self._slip_above = True
                self._burst_pending = True
            elif yaw_deviation_pct < _SLIP_OFF:
                self._slip_above = False

    def _make_burst(self) -> "np.ndarray":
        sr = self._sample_rate
        attack  = int(0.005 * sr)
        sustain = int(0.030 * sr)
        decay   = int(0.020 * sr)
        env = np.empty(attack + sustain + decay, dtype=np.float32)
        env[:attack] = np.linspace(0.0, 1.0, attack)
        env[attack:attack + sustain] = 1.0
        env[attack + sustain:] = np.linspace(1.0, 0.0, decay)
        noise = np.random.randn(len(env)).astype(np.float32)
        return noise * env * (self._volume * 2.0)

    def _audio_callback(self, outdata, frames, time_info, status):
        with self._lock:
            if self._muted:
                outdata[:] = 0.0
                return
            total_util    = self._total_util
            burst_pending = self._burst_pending
            self._burst_pending = False

        # Smooth glide: 0.3 per block toward target pitch
        self._current_freq += 0.3 * (_util_to_freq(total_util) - self._current_freq)
        omega = 2.0 * math.pi * self._current_freq / self._sample_rate
        sine = np.sin(self._phase + np.arange(frames, dtype=np.float64) * omega)
        sine = (sine * self._volume).astype(np.float32)
        self._phase = (self._phase + frames * omega) % (2.0 * math.pi)

        # Noise burst on rising edge of rear slip
        if burst_pending:
            self._burst_samples = self._make_burst()
            self._burst_pos = 0

        burst = np.zeros(frames, dtype=np.float32)
        if self._burst_samples is not None:
            remaining = len(self._burst_samples) - self._burst_pos
            n = min(remaining, frames)
            burst[:n] = self._burst_samples[self._burst_pos:self._burst_pos + n]
            self._burst_pos += n
            if self._burst_pos >= len(self._burst_samples):
                self._burst_samples = None

        if total_util > 100.0:
            over_pct = min((total_util - 100.0) / 30.0, 1.0)
            harmonic = np.sin(2 * (self._phase + np.arange(frames, dtype=np.float64) * omega))
            harmonic = (harmonic * self._volume * 0.15 * over_pct).astype(np.float32)
            out = np.clip(sine + burst + harmonic, -1.0, 1.0)
        else:
            out = np.clip(sine + burst, -1.0, 1.0)
        outdata[:, 0] = out
        outdata[:, 1] = out

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
