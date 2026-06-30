from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import irsdk

TELEMETRY_VARS = [
    "SessionTime",
    "Speed",
    "VelocityX",
    "VelocityY",
    "YawRate",
    "SteeringWheelAngle",
    "gLat",
    "gLong",
    # Alternative names used in some IBT versions
    "LatAccel",
    "LongAccel",
    # Roll used to subtract gravity from banking during lateral-velocity integration
    "Roll",
    "Throttle",
    "Brake",
    "LapDistPct",
    "Lap",
    "LapLastLapTime",
    # Wheel speeds — two naming conventions exist across IBT versions:
    #   WheelXXSpeed  rad/s  (older iRacing builds, requires × TYRE_RADIUS)
    #   XXspeed       m/s    (newer iRacing builds, linear surface speed)
    # tyre_energy_tracker detects which format is present automatically.
    "WheelRFSpeed",
    "WheelLFSpeed",
    "WheelRRSpeed",
    "WheelLRSpeed",
    "RFspeed",
    "LFspeed",
    "RRspeed",
    "LRspeed",
]

_G = 9.80665


def read_ibt(filepath: str) -> pd.DataFrame:
    """Read an IBT telemetry file. Returns DataFrame; metadata is in df.attrs."""
    ibt = irsdk.IBT()
    ibt.open(filepath)

    record_count: int = ibt._disk_header.session_record_count
    tick_rate: int = ibt._header.tick_rate
    available: set[str] = set(ibt.var_headers_names or [])

    data: dict[str, list] = {}
    for var in TELEMETRY_VARS:
        if var in available:
            data[var] = ibt.get_all(var)
        else:
            data[var] = [0.0] * record_count

    ibt.close()

    df = pd.DataFrame(data)
    df["time"] = (
        df["SessionTime"]
        if "SessionTime" in df.columns and df["SessionTime"].max() > 0
        else np.arange(record_count) / max(tick_rate, 1)
    )

    # ── Fix gLat / gLong ──────────────────────────────────────────────────────
    # Some IBT files expose lateral/longitudinal acceleration as LatAccel
    # (m/s²) rather than gLat (G).  Normalise to G units.
    if df["gLat"].abs().max() < 0.01 and "LatAccel" in available:
        df["gLat"] = df["LatAccel"] / _G
    if df["gLong"].abs().max() < 0.01 and "LongAccel" in available:
        df["gLong"] = df["LongAccel"] / _G

    # ── Vehicle-frame lateral velocity via dynamics integration ───────────────
    # VelocityX/Y from iRacing are world-frame components and cannot be used
    # directly as vehicle-frame lateral/longitudinal velocities.
    #
    # The correct lateral dynamics equation (Newton, vehicle frame):
    #   dv_lat/dt = LatAccel_vehicle − Speed × YawRate
    #
    # LatAccel is already in vehicle frame (lateral axis of car body).
    # Integrating this gives vehicle-frame lateral velocity at the CG.
    # Drift is managed by slow decay on straights and reset at standstill.
    lat_accel_ms2 = (df["LatAccel"].to_numpy(float) if "LatAccel" in available
                     else df["gLat"].to_numpy(float) * _G)
    spd = df["Speed"].to_numpy(float)
    yr = df["YawRate"].to_numpy(float)
    roll = df["Roll"].to_numpy(float) if "Roll" in available else np.zeros(record_count)
    dt = 1.0 / max(tick_rate, 1)

    # Subtract gravity component from body roll so integration doesn't drift
    # on banked sections.  Roll is the car-body roll angle (rad); the lateral
    # gravity projection is g·sin(Roll).
    lat_accel_pure = lat_accel_ms2 - _G * np.sin(roll)

    v_lat = np.zeros(record_count, dtype=float)
    for i in range(1, record_count):
        v_lat[i] = v_lat[i - 1] + (lat_accel_pure[i - 1] - spd[i - 1] * yr[i - 1]) * dt
        if spd[i] < 1.0:
            v_lat[i] = 0.0          # reset at standstill
        elif abs(lat_accel_pure[i]) < 0.5:
            v_lat[i] *= 0.999       # slow drift-to-zero on straights

    # Overwrite VelocityX with vehicle-frame lateral velocity and
    # VelocityY with forward speed so slip_calculator.py works correctly.
    df["VelocityX"] = v_lat
    df["VelocityY"] = spd

    # iRacing IBT filenames follow: {car_directory}_{track_name} {date}.ibt
    # Extract the car directory prefix and use it for car-override matching.
    stem = Path(filepath).stem
    car_slug = stem.split("_")[0] if "_" in stem else ""

    df.attrs["tick_rate"] = tick_rate
    df.attrs["record_count"] = record_count
    df.attrs["filepath"] = filepath
    df.attrs["car_name"] = car_slug
    df.attrs["available_vars"] = available

    return df


def get_latest_ibt(telemetry_dir: str) -> Optional[str]:
    p = Path(telemetry_dir)
    if not p.exists():
        return None
    files = list(p.rglob("*.ibt"))
    if not files:
        return None
    return str(max(files, key=lambda f: f.stat().st_mtime))


def find_ibt_files(telemetry_dir: str) -> list[str]:
    p = Path(telemetry_dir)
    if not p.exists():
        return []
    return sorted(
        (str(f) for f in p.rglob("*.ibt")),
        key=lambda f: Path(f).stat().st_mtime,
        reverse=True,
    )
