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

        # Filter states sized for mono (1 channel)
        n_ch = 1
        self._zi_hp = np.stack([sosfilt_zi(self._hp_sos)] * n_ch, axis=-1)
        self._zi_notch = [
            np.stack([sosfilt_zi(s)] * n_ch, axis=-1)
            for s in self._notch_filters
        ]
        self._zi_tyre = np.stack([sosfilt_zi(self._tyre_sos)] * n_ch, axis=-1)

        self._q    = queue.Queue(maxsize=4)
        self._lock = threading.Lock()

    def _process(self, x):
        """Process audio; x is (frames, channels) from sounddevice."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        else:
            x = x.T  # → (channels, frames); use first channel only
            x = x[:1]

        x = x.astype(np.float64)

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
