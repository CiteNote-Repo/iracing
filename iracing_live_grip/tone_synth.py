import threading

try:
    import numpy as np
    import sounddevice as sd
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_PULSE_VOL  = 0.25   # rear slide transient volume
_PULSE_DUR  = 0.08   # seconds — controls how long the transient gate stays open
_SLIP_ON    = 0.03   # rising threshold
_SLIP_OFF   = 0.02   # hysteresis reset


class GripToneSynth:
    """
    Stereo pink-noise engine with front-back HRTF spatial positioning.

    Base texture (overall tyre load) goes to both ears equally.
    Front scrub signal is processed through a frontal HRTF boost (4-6 kHz pinna
    resonance) so it appears to come from in front of the listener.
    Rear slide transient is processed through a rear HRTF notch (8-10 kHz pinna
    shadow) plus low-frequency emphasis so it appears to come from behind.
    A 5-sample ITD on the rear signal reinforces the back-of-head cue.
    Enable Windows Sonic or Dolby Atmos in Windows Sound Settings for best results.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        blocksize: int = 512,
        volume: float = 0.015,
    ):
        self._sample_rate = sample_rate
        self._blocksize = blocksize
        self._volume = volume

        # Shared state — written by telemetry thread, read by audio callback
        self._total_util: float = 0.0
        self._scrub_proximity: float = 100.0
        self._muted: bool = True
        self._slip_above: bool = False
        self._pulse_samples_remaining: int = 0
        self._lock = threading.Lock()

        # Audio callback thread only
        self._frame_offset: int = 0
        if _AVAILABLE:
            self._pink_state = np.zeros(8)
            # Pre-compute HRTF filter coefficients — fixed frequency so safe to cache
            from scipy.signal import butter
            nyq = sample_rate / 2
            self._front_sos      = butter(2, [3500/nyq, 7000/nyq],
                                          btype='band',     output='sos')
            self._rear_notch_sos = butter(2, [7500/nyq, min(11000/nyq, 0.99)],
                                          btype='bandstop', output='sos')
            self._rear_low_sos   = butter(2, 400/nyq,
                                          btype='low',      output='sos')

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
            self._total_util = total_util
            self._scrub_proximity = scrub_proximity_pct
            # Slip transient: fire once on rising edge, reset on hysteresis fall
            if rear_slip_raw > _SLIP_ON and not self._slip_above:
                self._slip_above = True
                self._pulse_samples_remaining = int(_PULSE_DUR * self._sample_rate)
            elif rear_slip_raw < _SLIP_OFF:
                self._slip_above = False

    def _pink_noise(self, frames: int) -> np.ndarray:
        """Generate pink noise (1/f spectrum) via Voss-McCartney algorithm.

        Pre-generates all random values in vectorized numpy calls and maintains
        a running sum to avoid calling state.sum() 512 times per block.
        """
        if not hasattr(self, '_pink_state'):
            self._pink_state = np.zeros(8)
        white = np.random.randn(frames)
        idxs  = np.random.randint(0, 8, frames)
        vals  = np.random.randn(frames)
        out   = np.empty(frames)
        state = self._pink_state
        running_sum = state.sum()
        for i in range(frames):
            running_sum += vals[i] - state[idxs[i]]
            state[idxs[i]] = vals[i]
            out[i] = white[i] + running_sum
        peak = np.abs(out).max()
        if peak > 0:
            out /= peak
        return out

    def _shape_spectrum(self, noise: np.ndarray, total_util: float) -> np.ndarray:
        from scipy.signal import sosfilt
        sr = self._sample_rate
        if total_util < 70:
            low, high = 200, 1500
            gain = 0.15                           # near-silent whisper
        elif total_util < 90:
            ratio = (total_util - 70) / 20
            low, high = 200, int(2000 + ratio * 6000)
            gain = 0.15 + ratio * 0.85            # ramps to full
        elif total_util <= 100:
            low, high = 500, 8000
            gain = 1.0                            # full zipper texture
        else:
            over = min((total_util - 100) / 30, 1.0)
            low, high = 200, int(8000 - over * 6500)
            gain = 1.0 - over * 0.3              # slightly reduced past limit
        nyq = sr / 2
        high = min(high, int(nyq * 0.95))
        low = max(low, 20)
        if not hasattr(self, '_sos_cache') or self._sos_cache[0] != (low, high):
            from scipy.signal import butter
            sos = butter(2, [low / nyq, high / nyq], btype='band', output='sos')
            self._sos_cache = ((low, high), sos)
        else:
            sos = self._sos_cache[1]
        return sosfilt(sos, noise) * gain

    def _apply_scrub_modulation(
        self,
        signal: np.ndarray,
        scrub_proximity: float,
        frame_offset: int,
    ) -> np.ndarray:
        # 0 means no data (gates not met) — never modulate in this case
        if scrub_proximity <= 0 or scrub_proximity >= 70:
            return signal
        depth = (70 - scrub_proximity) / 70
        t = (np.arange(len(signal)) + frame_offset) / self._sample_rate
        modulator = 1.0 - depth * 0.4 * (0.5 + 0.5 * np.sin(2 * np.pi * 60 * t))
        return signal * modulator

    def _apply_front_hrtf(self, signal: np.ndarray) -> np.ndarray:
        """Approximate HRTF for sound positioned in front.
        Boost 4-6kHz (pinna resonance for frontal sounds), applied to both channels.
        """
        from scipy.signal import sosfilt
        boost = sosfilt(self._front_sos, signal) * 0.4
        return signal + boost

    def _apply_rear_hrtf(self, signal: np.ndarray) -> np.ndarray:
        """Approximate HRTF for sound positioned behind.
        Notch at 8-10kHz (pinna shadowing) plus low-frequency emphasis.
        """
        from scipy.signal import sosfilt
        signal = sosfilt(self._rear_notch_sos, signal)
        boost = sosfilt(self._rear_low_sos, signal) * 0.3
        return signal + boost

    def _rear_slide_transient(self, frames: int) -> np.ndarray:
        sr = self._sample_rate
        attack_s  = int(0.002 * sr)
        sustain_s = int(0.040 * sr)
        decay_s   = int(0.025 * sr)
        total_s   = attack_s + sustain_s + decay_s
        envelope = np.zeros(total_s)
        envelope[:attack_s] = np.linspace(0, 1, attack_s)
        envelope[attack_s:attack_s + sustain_s] = 1.0
        envelope[attack_s + sustain_s:] = np.linspace(1, 0, decay_s)
        burst = np.random.randn(total_s) * envelope * _PULSE_VOL
        out = np.zeros(frames)
        copy_len = min(total_s, frames)
        out[:copy_len] = burst[:copy_len]
        return out

    def _audio_callback(self, outdata, frames, time_info, status):
        with self._lock:
            if self._muted:
                outdata[:] = 0.0
                return
            total_util    = self._total_util
            scrub_prox    = self._scrub_proximity
            vol           = self._volume
            pulse_samples = min(self._pulse_samples_remaining, frames)
            self._pulse_samples_remaining -= pulse_samples

        # Base texture: pink noise shaped by total utilisation → both ears equally
        noise  = self._pink_noise(frames)
        shaped = self._shape_spectrum(noise, total_util)
        shaped = shaped.astype(np.float32) * vol

        # Front scrub signal: AM modulation when scrub_prox < 70%
        # Processed through front HRTF — appears to come from in front
        front_signal      = self._apply_scrub_modulation(shaped.copy(), scrub_prox, self._frame_offset)
        front_spatialized = self._apply_front_hrtf(front_signal)

        # Rear slide signal: noise burst when rear slip detected
        # Processed through rear HRTF — appears to come from behind
        rear_signal = shaped.copy()
        if pulse_samples > 0:
            transient = self._rear_slide_transient(frames)
            rear_signal[:pulse_samples] += transient[:pulse_samples].astype(np.float32)
        rear_spatialized = self._apply_rear_hrtf(rear_signal)

        # Mix: base texture in both channels, spatial signals additive on top
        left  = shaped * 0.5 + front_spatialized * 0.4 + rear_spatialized * 0.2
        right = shaped * 0.5 + front_spatialized * 0.4 + rear_spatialized * 0.2

        # 5-sample ITD on rear signal: left ear receives it slightly later,
        # reinforcing the rear-of-head spatial cue
        itd_samples = 5
        if len(rear_spatialized) > itd_samples:
            left  += np.pad(rear_spatialized, (itd_samples, 0))[:frames] * 0.15
            right += rear_spatialized * 0.15

        outdata[:, 0] = np.clip(left,  -1.0, 1.0).astype(np.float32)
        outdata[:, 1] = np.clip(right, -1.0, 1.0).astype(np.float32)
        self._frame_offset += frames

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
