#!/usr/bin/env python3
"""
iRacing Live Grip Monitor — real-time bilateral axle utilisation overlay + audio.

Usage:
  python main.py              # live mode, connects to iRacing
  python main.py --test       # simulated corner sequence, no iRacing needed
  python main.py --no-audio   # visual only — use this first for visual calibration
  python main.py --car "Acura NSX GT3 EVO22"   # car-specific peak_glat_g
  python main.py --blocksize 256               # lower audio latency
"""

import argparse
import math
import sys
import threading
import time
from typing import Optional


# ── dependency check ──────────────────────────────────────────────────────────

def _check_deps(audio: bool) -> None:
    missing = []
    try:
        import irsdk  # noqa: F401
    except ImportError:
        missing.append("pyirsdk")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")
    if audio:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            missing.append("sounddevice")
    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────

from config import load_config, update_overlay_position, resolve_peak_glat
from grip_calculator import GripData, classify_acoustic_state
from live_telemetry import LiveTelemetry
from tone_synth import GripToneSynth
from overlay import GripOverlay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="iRacing Live Grip Monitor")
    p.add_argument("--test", action="store_true",
                   help="Simulated corner sequence — no iRacing needed")
    p.add_argument("--no-audio", action="store_true",
                   help="Visual overlay only, no audio tone")
    p.add_argument("--car", default="",
                   help="Car name for peak_glat_g lookup (e.g. 'Ferrari 296 GT3')")
    p.add_argument("--blocksize", type=int, default=512,
                   help="Audio blocksize in samples (default 512; try 256 or 128 "
                        "for lower latency — watch for iRacing frame stutters)")
    return p.parse_args()


# ── test mode ─────────────────────────────────────────────────────────────────

# Each tuple: (time_s, front_util_pct, rear_util_pct)
_TEST_PHASES = [
    (0.0,    0.0,   0.0),
    (2.0,    0.0,   0.0),
    (4.0,   90.0,   5.0),
    (6.0,  105.0, 105.0),
    (8.0,   10.0, 115.0),
    (10.0,   0.0,   0.0),
]

_TEST_ANNOUNCE = [
    (0.0,  "[0-2s]   Both quiet — low tone (110 Hz both channels)"),
    (2.0,  "[2-4s]   Front rising — left channel climbs, right stays low"),
    (4.0,  "[4-6s]   Both rising — channels converge toward unison (BILATERAL STATE)"),
    (6.0,  "[6-8s]   Rear exceeds 100% — right channel crosses into HOWL range"),
    (8.0,  "[8-10s]  Both decay back to quiet"),
]


def _interp(phases, t: float) -> tuple:
    """Linear interpolation between test phase keyframes."""
    t = t % phases[-1][0]
    for i in range(len(phases) - 1):
        t0, f0, r0 = phases[i]
        t1, f1, r1 = phases[i + 1]
        if t0 <= t < t1:
            frac = (t - t0) / (t1 - t0)
            return f0 + frac * (f1 - f0), r0 + frac * (r1 - r0)
    return phases[-1][1], phases[-1][2]


def run_test_mode(
    telem: LiveTelemetry,
    synth: Optional[GripToneSynth],
    min_speed_for_audio: float,
) -> None:
    print("\nTest mode — 10-second corner sequence (cycles continuously):")
    for _, msg in _TEST_ANNOUNCE:
        print(f"  {msg}")
    print()

    start = time.time()
    last_phase = -1
    cycle = _TEST_PHASES[-1][0]

    while True:
        elapsed = time.time() - start
        t = elapsed % cycle
        phase_idx = sum(1 for ts, *_ in _TEST_PHASES if ts <= t) - 1

        if phase_idx != last_phase:
            if 0 <= phase_idx < len(_TEST_ANNOUNCE):
                print(f"  {_TEST_ANNOUNCE[phase_idx][1]}")
            last_phase = phase_idx

        front_util, rear_util = _interp(_TEST_PHASES, t)
        total_util = max(front_util, rear_util)

        # Inject simulated rear slip only when rear is meaningfully over-limit
        rear_slip = 0.06 if rear_util > 90.0 else None

        front_state = classify_acoustic_state(front_util)
        rear_state = classify_acoustic_state(rear_util, rear_slip=rear_slip)
        overall_state = classify_acoustic_state(
            total_util, rear_slip=rear_slip
        )

        data = GripData(
            front_util=front_util,
            rear_util=rear_util,
            total_util=total_util,
            front_state=front_state,
            rear_state=rear_state,
            overall_state=overall_state,
            is_on_track=True,
            speed_mps=100.0,
            connected=True,
        )
        telem._set_grip(data)

        if synth:
            active = data.speed_mps >= min_speed_for_audio
            synth.set_utilization(front_util, rear_util, active=active)

        time.sleep(1.0 / 60.0)


# ── live iRacing loop ─────────────────────────────────────────────────────────

def run_live(
    telem: LiveTelemetry,
    synth: Optional[GripToneSynth],
    min_speed_for_audio: float,
) -> None:
    import irsdk

    ir = irsdk.IRSDK()
    print("Waiting for iRacing to start...")

    def on_update(data: GripData) -> None:
        if synth:
            active = data.connected and data.is_on_track and data.speed_mps >= min_speed_for_audio
            synth.set_utilization(data.front_util, data.rear_util, active=active)

    telem.run(ir, on_update=on_update)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    want_audio = not args.no_audio

    if not args.test:
        _check_deps(audio=want_audio)

    cfg = load_config()
    peak_glat_g = resolve_peak_glat(cfg, args.car)

    if args.car:
        print(f"Car: {args.car!r} → peak_glat_g = {peak_glat_g:.2f} G")
    else:
        print(f"peak_glat_g = {peak_glat_g:.2f} G  (set --car or edit live_grip_config.json)")

    # ── audio ──
    synth: Optional[GripToneSynth] = None
    if want_audio:
        if not GripToneSynth.available():
            print("sounddevice not installed — running visual only. "
                  "Install with: pip install sounddevice")
        else:
            try:
                synth = GripToneSynth(
                    blocksize=args.blocksize,
                    volume=float(cfg.get("audio_volume", 0.15)),
                )
                synth.start()
                print(f"Audio: ON  (blocksize={args.blocksize}, "
                      f"vol={cfg.get('audio_volume', 0.15):.2f})")
            except Exception as e:
                print(f"Audio failed to start: {e}\nContinuing with visual only.")
                synth = None
    else:
        print("Audio: OFF  (--no-audio)")

    # ── telemetry provider ──
    telem = LiveTelemetry(peak_glat_g)
    min_speed = float(cfg.get("min_speed_for_audio", 5.0))

    # ── overlay ──
    def on_drag_end(x: int, y: int) -> None:
        update_overlay_position(cfg, x, y)

    overlay = GripOverlay(cfg, telem.get_grip, on_drag_end=on_drag_end)

    # ── background thread ──
    if args.test:
        bg = threading.Thread(
            target=run_test_mode,
            args=(telem, synth, min_speed),
            daemon=True,
        )
    else:
        bg = threading.Thread(
            target=run_live,
            args=(telem, synth, min_speed),
            daemon=True,
        )

    bg.start()

    # Overlay runs on main thread (tkinter requirement)
    try:
        overlay.run()
    finally:
        if synth:
            synth.stop()


if __name__ == "__main__":
    main()
