"""
Tyre slip and acoustic state estimation from iRacing wheel-speed telemetry.

Two wheel-speed naming conventions exist across IBT versions:
  WheelXXSpeed  rad/s  — older iRacing builds; surface speed = value × TYRE_RADIUS
  XXspeed       m/s    — newer iRacing builds; already linear surface speed

Both are detected automatically.  When neither is present (all zeros),
the module falls back to lateral_utilization_pct-only classification —
REAR_SLIDE and FRONT_LOCK require slip data and will not appear in that mode.

Acoustic state priority (highest wins when conditions overlap):
  OVERSTEER_HOWL  >  REAR_SLIDE / FRONT_LOCK  >  PEAK_GRIP_HISS  >  QUIET
"""

import numpy as np
import pandas as pd

TYRE_RADIUS = 0.33  # metres — GT3 approximate (used for rad/s → m/s conversion)

# Integer state constants
QUIET          = 0
PEAK_GRIP_HISS = 1
REAR_SLIDE     = 2
FRONT_LOCK     = 3
OVERSTEER_HOWL = 4

STATE_NAMES: dict[int, str] = {
    QUIET:          "Quiet",
    PEAK_GRIP_HISS: "Peak Grip Hiss",
    REAR_SLIDE:     "Rear Slide",
    FRONT_LOCK:     "Front Lock",
    OVERSTEER_HOWL: "Oversteer Howl",
}

STATE_COLORS: dict[int, str] = {
    QUIET:          "rgba(44,62,80,0.0)",
    PEAK_GRIP_HISS: "rgba(46,204,113,0.12)",
    REAR_SLIDE:     "rgba(230,126,34,0.28)",
    FRONT_LOCK:     "rgba(52,152,219,0.25)",
    OVERSTEER_HOWL: "rgba(231,76,60,0.38)",
}


def _get_wheel_surfaces(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Return (rear_surface_mps, front_surface_mps, available).

    Tries formats in priority order:
      1. WheelRRSpeed / WheelLRSpeed / WheelRFSpeed / WheelLFSpeed  (rad/s)
         → multiply by TYRE_RADIUS to get m/s
      2. RRspeed / LRspeed / RFspeed / LFspeed  (m/s, newer IBT format)
         → use directly
    Returns zeros and available=False when neither format has non-zero data.
    """
    n = len(df)

    # Format 1: rad/s (WheelXXSpeed)
    rads_cols = ("WheelRRSpeed", "WheelLRSpeed", "WheelRFSpeed", "WheelLFSpeed")
    if all(c in df.columns for c in rads_cols):
        rr = df["WheelRRSpeed"].to_numpy(float)
        lr = df["WheelLRSpeed"].to_numpy(float)
        rf = df["WheelRFSpeed"].to_numpy(float)
        lf = df["WheelLFSpeed"].to_numpy(float)
        if max(float(np.abs(rr).max()), float(np.abs(lr).max()),
               float(np.abs(rf).max()), float(np.abs(lf).max())) > 0.1:
            return (
                (rr + lr) / 2.0 * TYRE_RADIUS,
                (rf + lf) / 2.0 * TYRE_RADIUS,
                True,
            )

    # Format 2: m/s (XXspeed — newer builds)
    mps_cols = ("RRspeed", "LRspeed", "RFspeed", "LFspeed")
    if all(c in df.columns for c in mps_cols):
        rr = df["RRspeed"].to_numpy(float)
        lr = df["LRspeed"].to_numpy(float)
        rf = df["RFspeed"].to_numpy(float)
        lf = df["LFspeed"].to_numpy(float)
        if max(float(np.abs(rr).max()), float(np.abs(lr).max()),
               float(np.abs(rf).max()), float(np.abs(lf).max())) > 0.1:
            return (rr + lr) / 2.0, (rf + lf) / 2.0, True

    return np.zeros(n), np.zeros(n), False


def compute_tyre_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rear_slip, front_slip, acoustic_state, and wheel_speeds_available columns.

    Prerequisite: lateral_utilization_pct must already be in df
    (from compute_absolute_utilization).
    """
    df = df.copy()
    spd = df["Speed"].to_numpy(float)

    rear_surface, front_surface, wheel_available = _get_wheel_surfaces(df)
    safe_spd = np.maximum(spd, 1.0)

    df["wheel_speeds_available"] = wheel_available
    if wheel_available:
        df["rear_slip"]  = (rear_surface  - spd) / safe_spd
        df["front_slip"] = (front_surface - spd) / safe_spd
    else:
        df["rear_slip"]  = np.zeros(len(df))
        df["front_slip"] = np.zeros(len(df))

    # ── Acoustic state classification ─────────────────────────────────────────
    util = df["lateral_utilization_pct"].to_numpy(float)
    rs   = df["rear_slip"].to_numpy(float)
    fs   = df["front_slip"].to_numpy(float)

    state = np.full(len(df), QUIET, dtype=np.int8)

    state[util >= 75] = PEAK_GRIP_HISS
    if wheel_available:
        state[(util > 90) & (rs > 0.05)]  = REAR_SLIDE
        state[(util > 90) & (fs < -0.05)] = FRONT_LOCK
        state[(util > 100) & (rs > 0.08)] = OVERSTEER_HOWL
    else:
        state[util > 100] = OVERSTEER_HOWL
    state[util < 75] = QUIET

    df["acoustic_state"] = state.astype(int)
    return df
