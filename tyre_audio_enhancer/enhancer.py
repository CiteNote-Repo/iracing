import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt_zi, sosfilt

SAMPLE_RATE = 48000
BLOCKSIZE   = 256  # ~5ms latency

class TyreAudioEnhancer:
    def __init__(self, input_device, output_device,
                 engine_cut_db=-20, tyre_boost_db=12,
                 notch_freqs=None):
        self.sr = SAMPLE_RATE

        if notch_freqs is None:
            notch_freqs = [200, 320, 400, 600, 800]

        # High-pass to remove low-frequency engine rumble below 120Hz
        self._hp_sos = butter(
            4, 120/(self.sr/2), btype='high', output='sos'
        )

        # Notch filters for each engine harmonic frequency
        self._notch_filters = []
        for f in notch_freqs:
            bw = f * 0.15
            low  = max(0.001, (f - bw/2) / (self.sr/2))
            high = min(0.999, (f + bw/2) / (self.sr/2))
            if low < high:
                sos = butter(2, [low, high],
                            btype='bandstop', output='sos')
                self._notch_filters.append(sos)

        # Bandpass for tyre-dominant frequencies (800Hz-8kHz)
        self._tyre_sos = butter(
            2, [800/(self.sr/2), 8000/(self.sr/2)],
            btype='band', output='sos'
        )

        self._engine_gain = 10**(engine_cut_db/20)
        self._tyre_gain   = 10**(tyre_boost_db/20)

        # Initialize stateful filter conditions for stereo (2 channels)
        self._zi_hp = np.stack(
            [sosfilt_zi(self._hp_sos)] * 2, axis=-1
        )
        self._zi_notch = [
            np.stack([sosfilt_zi(s)] * 2, axis=-1)
            for s in self._notch_filters
        ]
        self._zi_tyre = np.stack(
            [sosfilt_zi(self._tyre_sos)] * 2, axis=-1
        )

        self._input_device  = input_device
        self._output_device = output_device

    def _process_block(self, x):
        # x shape: (2, frames) — stereo, float32

        # Remove sub-120Hz engine rumble
        x, self._zi_hp = sosfilt(
            self._hp_sos, x, zi=self._zi_hp
        )

        # Apply engine harmonic notches
        for i, sos in enumerate(self._notch_filters):
            x, self._zi_notch[i] = sosfilt(
                sos, x, zi=self._zi_notch[i]
            )

        # Extract tyre band and boost it
        x_tyre, self._zi_tyre = sosfilt(
            self._tyre_sos, x, zi=self._zi_tyre
        )

        # Mix: notched signal + extra tyre band gain
        x_out = x + x_tyre * (self._tyre_gain - 1.0)

        # Soft clip to prevent distortion
        x_out = np.tanh(x_out * 0.8) / 0.8

        return x_out

    def run(self):
        def callback(indata, outdata, frames, time, status):
            x = indata.T.astype(np.float64)
            processed = self._process_block(x)
            outdata[:] = processed.T.astype(np.float32)

        print(f"Input:  {self._input_device}")
        print(f"Output: {self._output_device}")
        print("Tyre Audio Enhancer running — Ctrl+C to stop")

        with sd.Stream(
            device=(self._input_device, self._output_device),
            samplerate=self.sr,
            blocksize=BLOCKSIZE,
            dtype='float32',
            channels=2,
            callback=callback
        ):
            sd.sleep(999_999_999)
