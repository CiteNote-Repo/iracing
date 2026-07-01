#!/usr/bin/env python3
"""
iRacing Fuel Calculator — automatic pit fuel setter with transparent overlay.
Usage:
  python main.py          # normal mode (connects to iRacing)
  python main.py --test   # simulated race, no iRacing needed
"""

import sys
import threading
import time
import argparse

# ── dependency check ──────────────────────────────────────────────────────────
def _check_deps():
    missing = []
    try:
        import irsdk  # noqa: F401
    except ImportError:
        missing.append("pyirsdk")
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    if missing:
        print("Missing dependencies. Install with:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────

from config import load_config, save_config, update_overlay_position
from fuel_tracker import FuelTracker, WAITING
from overlay import FuelOverlay
from pit_controller import PitController
from logger import log_info, log_error


def parse_args():
    p = argparse.ArgumentParser(description="iRacing Fuel Calculator")
    p.add_argument("--test", action="store_true", help="Run with simulated data (no iRacing)")
    p.add_argument("--alpha", type=float, default=None,
                   help="Overlay opacity 0.0-1.0 (overrides config)")
    return p.parse_args()


# ── Test mode simulation ──────────────────────────────────────────────────────

def run_test_mode(tracker: FuelTracker, pit_ctrl: PitController) -> None:
    """Simulate a 40-min race, 20 min elapsed, 2.3 L/lap, pit entry at T+10s."""
    log_info("TEST MODE: simulating 40-min race, 20 min remaining, 2.3L/lap")

    lap_time = 90.0       # 1:30 per lap
    fuel_per_lap = 2.3
    total_race = 2400.0   # 40 min
    elapsed = 1200.0      # 20 min elapsed
    time_remain = total_race - elapsed
    current_fuel = 4.5
    lap_num = int(elapsed / lap_time)

    # Seed 5 clean laps of history
    for i in range(1, 6):
        telem = {
            "FuelLevel": current_fuel + fuel_per_lap * (6 - i),
            "FuelLevelPct": (current_fuel + fuel_per_lap * (6 - i)) / 60.0,
            "Lap": i,
            "LapLastLapTime": lap_time + (0.1 * (i - 3)),
            "SessionTimeRemain": time_remain + (6 - i) * lap_time,
            "SessionLapsRemain": 9999,
            "SessionType": "Race",
            "IsOnTrack": True,
            "OnPitRoad": False,
        }
        tracker.update(telem)
        time.sleep(0.05)

    # Advance to "current lap"
    telem_base = {
        "FuelLevel": current_fuel,
        "FuelLevelPct": current_fuel / 60.0,
        "Lap": lap_num,
        "LapLastLapTime": lap_time,
        "SessionTimeRemain": time_remain,
        "SessionLapsRemain": 9999,
        "SessionType": "Race",
        "IsOnTrack": True,
        "OnPitRoad": False,
    }
    tracker.update(telem_base)

    log_info("TEST MODE: 10 seconds until simulated pit entry...")
    time.sleep(10)

    # Trigger pit entry
    log_info("TEST MODE: simulating pit entry")
    pit_telem = dict(telem_base)
    pit_telem["OnPitRoad"] = True
    fuel_cmd = tracker.update(pit_telem)

    if fuel_cmd is not None and fuel_cmd > 0:
        pit_ctrl.handle_pit_entry(fuel_cmd, time_remain)
        log_info(f"TEST MODE: pit entry handled, {fuel_cmd:.1f}L commanded")
    else:
        log_info("TEST MODE: pit entry detected but no fuel commanded (insufficient data?)")

    # Stay on pit road briefly then continue simulating
    for _ in range(5):
        tracker.update(pit_telem)
        time.sleep(1)

    log_info("TEST MODE: simulation complete, overlay stays open")


# ── Live iRacing loop ─────────────────────────────────────────────────────────

def run_live(tracker: FuelTracker, pit_ctrl: PitController) -> None:
    import irsdk

    ir = irsdk.IRSDK()
    pit_ctrl._ir = ir
    connected = False

    print("Waiting for iRacing to start...")
    log_info("Starting live telemetry loop")

    while True:
        try:
            if not ir.is_initialized or not ir.is_connected:
                if connected:
                    log_info("iRacing disconnected — reconnecting...")
                    connected = False
                    tracker.state.state = WAITING

                ir.startup()
                if ir.is_initialized and ir.is_connected:
                    connected = True
                    log_info("Connected to iRacing")
                else:
                    time.sleep(2)
                    continue

            ir.freeze_var_buffer_latest()
            telem = {
                "FuelLevel": ir["FuelLevel"] or 0.0,
                "FuelLevelPct": ir["FuelLevelPct"] or 0.0,
                "Lap": ir["Lap"] or 0,
                "LapLastLapTime": ir["LapLastLapTime"] or 0.0,
                "SessionTimeRemain": ir["SessionTimeRemain"] or 0.0,
                "SessionLapsRemain": ir["SessionLapsRemainEx"] or ir["SessionLapsRemain"] or 9999,
                "SessionType": ir["SessionType"] or "",
                "IsOnTrack": bool(ir["IsOnTrack"]),
                "OnPitRoad": bool(ir["OnPitRoad"]),
            }

            fuel_cmd = tracker.update(telem)
            if fuel_cmd is not None and fuel_cmd > 0:
                pit_ctrl.handle_pit_entry(fuel_cmd, telem["SessionTimeRemain"])

        except Exception as e:
            log_error(f"Telemetry loop error: {e}")
            time.sleep(2)

        time.sleep(1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.test:
        _check_deps()

    cfg = load_config()
    if args.alpha is not None:
        cfg["overlay_alpha"] = max(0.0, min(1.0, args.alpha))

    tracker = FuelTracker(cfg)

    def on_drag_end(x, y):
        update_overlay_position(cfg, x, y)

    overlay = FuelOverlay(cfg, tracker.get_state, on_drag_end=on_drag_end)
    pit_ctrl = PitController(ir=None, fuel_tracker=tracker, overlay_ref=overlay)

    # Start background telemetry thread
    if args.test:
        telem_thread = threading.Thread(
            target=run_test_mode, args=(tracker, pit_ctrl), daemon=True
        )
    else:
        telem_thread = threading.Thread(
            target=run_live, args=(tracker, pit_ctrl), daemon=True
        )

    telem_thread.start()

    # Run overlay on main thread (tkinter requirement)
    log_info("Overlay starting")
    overlay.run()
    log_info("Overlay closed — exiting")


if __name__ == "__main__":
    main()
