import queue
import threading

import sounddevice as sd
import numpy as np
from scipy.signal import butter, sosfilt_zi, sosfilt


class TyreAudioEnhancer:
    def __init__(self, input_device, output_device,
                 engine_cut_db=-20, tyre_boost_db=12,
                 notch_freqs=None):

        self._input_device  = input_device
        self._output_device = output_device

        in_info  = sd.query_devices(input_device)
        out_info = sd.query_devices(output_device)

        self._in_channels  = min(2, int(in_info['max_input_channels']))
        self._out_channels = min(2, int(out_info['max_output_channels']))
        self._in_sr  = int(in_info['default_samplerate'])
        self._out_sr = int(out_info['default_samplerate'])
        self._blocksize = 512

        print(f"Input:  device {input_device}, "
              f"{self._in_channels}ch, {self._in_sr}Hz")
        print(f"Output: device {output_device}, "
              f"{self._out_channels}ch, {self._out_sr}Hz")

        if notch_freqs is None:
            notch_freqs = [190, 270, 340, 550, 790]

        nyq = self._in_sr / 2

        self._hp_sos = butter(4, 120/nyq, btype='high', output='sos')

        self._notch_filters = []
        for f in notch_freqs:
            bw = f * 0.15
            low  = max(0.001, (f - bw/2) / nyq)
            high = min(0.999, (f + bw/2) / nyq)
            if low < high:
                sos = butter(2, [low, high], btype='bandstop', output='sos')
                self._notch_filters.append(sos)

        self._tyre_sos = butter(
            2, [min(800/nyq, 0.9), min(8000/nyq, 0.99)],
            btype='band', output='sos'
        )

        self._tyre_gain = 10**(tyre_boost_db/20)

        # Filter states — shape must be (n_sections, n_channels, 2)
        # For mono: n_channels=1, so shape is (n_sections, 1, 2)
        def _make_zi(sos):
            zi_1ch = sosfilt_zi(sos)       # shape (n_sections, 2)
            return zi_1ch[:, np.newaxis, :]  # shape (n_sections, 1, 2)

        self._zi_hp    = _make_zi(self._hp_sos)
        self._zi_notch = [_make_zi(s) for s in self._notch_filters]
        self._zi_tyre  = _make_zi(self._tyre_sos)

        self._q    = queue.Queue(maxsize=4)
        self._lock = threading.Lock()

    def _process(self, x):
        """Process audio; x is (frames, channels) from sounddevice."""
        # Ensure shape is (1, frames) for mono
        if x.ndim == 1:
            x = x[np.newaxis, :]         # (1, frames)
        elif x.shape[0] > x.shape[1]:
            x = x.T                       # was (frames, 1), now (1, frames)

        x = x[:1].astype(np.float64)     # keep only first channel

        x, self._zi_hp = sosfilt(self._hp_sos, x, zi=self._zi_hp)

        for i, sos in enumerate(self._notch_filters):
            x, self._zi_notch[i] = sosfilt(sos, x, zi=self._zi_notch[i])

        x_tyre, self._zi_tyre = sosfilt(self._tyre_sos, x, zi=self._zi_tyre)

        x_out = x + x_tyre * (self._tyre_gain - 1.0)
        x_out = np.tanh(x_out * 0.8) / 0.8

        return x_out  # shape (1, frames)

    def run(self):
        def input_callback(indata, frames, time, status):
            try:
                processed = self._process(indata.copy())
                self._q.put_nowait(processed)
            except queue.Full:
                pass

        def output_callback(outdata, frames, time, status):
            try:
                data = self._q.get_nowait()   # shape (1, frames)
                stereo = np.vstack([data, data]).T  # → (frames, 2)
                n = min(stereo.shape[0], frames)
                outdata[:n] = stereo[:n].astype(np.float32)
                if n < frames:
                    outdata[n:] = 0
            except queue.Empty:
                outdata[:] = 0

        print("Tyre Audio Enhancer running — Ctrl+C to stop")
        print("(Audio processing: mono input → stereo output)")

        in_stream = sd.InputStream(
            device=self._input_device,
            samplerate=self._in_sr,
            blocksize=self._blocksize,
            channels=self._in_channels,
            dtype='float32',
            callback=input_callback,
        )
        out_stream = sd.OutputStream(
            device=self._output_device,
            samplerate=self._out_sr,
            blocksize=self._blocksize,
            channels=self._out_channels,
            dtype='float32',
            callback=output_callback,
        )

        with in_stream, out_stream:
            sd.sleep(999_999_999)
