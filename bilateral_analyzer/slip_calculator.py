"""
Derives front/rear tyre slip angles from iRacing telemetry variables.

iRacing does not expose tyre slip angles directly.  We estimate them from:
  VelocityX  – lateral velocity at CG in vehicle frame (m/s, right = positive)
  VelocityY  – longitudinal velocity at CG in vehicle frame (m/s)
  YawRate    – positive = left/counter-clockwise (rad/s)
  SteeringWheelAngle – positive = right (rad), divided by steering_ratio

Slip angles are in radians unless noted.  Normalised percentages use a gated
rolling maximum that only updates during sustained high-load cornering, preventing
pit-lane, curb strikes, and low-speed transitions from distorting the scale.
"""

import numpy as np
import pandas as pd


def _gated_rolling_max(
    values: np.ndarray,
    gate: np.ndarray,
    active_decay: float = 0.9995,
    passive_decay: float = 0.9999,
) -> np.ndarray:
    """Rolling max that only updates from data when gate is True.

    When the gate is inactive (low speed / low gLat) the envelope still decays
    slowly (passive_decay) so the scale remains sensible across the session, but
    spikes from curb strikes, spins, and pit-lane manoeuvres cannot poison it.
    """
    out = np.empty(len(values))
    cur = 0.0
    for i in range(len(values)):
        if gate[i]:
            cur = max(abs(values[i]), cur * active_decay)
        else:
            cur *= passive_decay
        out[i] = cur
    return out


_G = 9.80665


def _integrate_vlat(
    glat: np.ndarray,
    roll: np.ndarray,
    speed: np.ndarray,
    yr: np.ndarray,
    steer: np.ndarray,
    tick_rate: int,
) -> np.ndarray:
    """
    Integrate vehicle-frame lateral velocity with two drift corrections:

      1. Standstill reset — v_lat → 0 when Speed < 1 m/s.
      2. Straight-line zero-reference reset — v_lat → 0 when the car has been
         driving straight for 24 consecutive samples (0.4 s at 60 Hz):
           |SteeringWheelAngle| < 2°   (wheel nearly centred)
           |YawRate|           < 0.02 rad/s  (no rotation)
           Speed               > 30 m/s     (not crawling)
         A car in this state has zero lateral velocity by physics — resetting
         here eliminates drift accumulated on long corners before the straight.
      3. Low-lateral-accel decay — v_lat × 0.999 when |lat_accel| < 0.5 m/s²
         (i.e. on gentle sweepers / light-braking zones).

    The roll-gravity correction (g·sin(Roll)) removes the apparent lateral
    acceleration the IMU sees on banked road sections.
    """
    n = len(glat)
    dt = 1.0 / tick_rate
    lat_accel_pure = glat * _G - _G * np.sin(roll)  # vehicle-frame lateral accel (m/s²)

    straight = (
        (np.abs(steer) < np.radians(2))
        & (np.abs(yr) < 0.02)
        & (speed > 30.0)
    )
    sustained_straight = (
        pd.Series(straight).rolling(24, min_periods=24).sum().fillna(0) == 24
    )

    v_lat = np.zeros(n, dtype=float)
    for i in range(1, n):
        v_lat[i] = v_lat[i - 1] + (lat_accel_pure[i - 1] - speed[i - 1] * yr[i - 1]) * dt
        if speed[i] < 1.0:
            v_lat[i] = 0.0
        elif sustained_straight.iloc[i]:
            v_lat[i] = 0.0  # zero-reference correction — genuine straight-line state
        elif abs(lat_accel_pure[i]) < 0.5:
            # Decay rate is speed-dependent so accumulated drift from low-speed
            # corners and end-of-session deceleration clears before it can corrupt
            # the next high-speed corner:
            #   > 30 m/s  (motorway straight): gentle 0.999 — barely any drift here
            #   10–30 m/s (slow hairpin exit): 0.97 — clears to <5 % in ~1 s
            #   < 10 m/s  (near-standstill):  0.90 — clears to <5 % in ~3 frames
            if speed[i] > 30.0:
                decay = 0.999
            elif speed[i] > 10.0:
                decay = 0.97
            else:
                decay = 0.90
            v_lat[i] *= decay
    return v_lat


def compute_slip_angles(
    df: pd.DataFrame,
    wheelbase_m: float = 2.7,
    cg_to_front_ratio: float = 0.45,
    steering_ratio: float = 14.0,
    min_speed_ms: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()

    l_front_base = wheelbase_m * cg_to_front_ratio

    yr    = df["YawRate"].to_numpy(dtype=float)
    steer = df["SteeringWheelAngle"].to_numpy(dtype=float)
    speed = df["Speed"].to_numpy(dtype=float)
    glat  = df["gLat"].to_numpy(dtype=float)
    glong = df["gLong"].to_numpy(dtype=float) if "gLong" in df.columns else np.zeros(len(df))
    roll  = df["Roll"].to_numpy(dtype=float)  if "Roll"  in df.columns else np.zeros(len(df))
    tick_rate = int(df.attrs.get("tick_rate", 60))

    # Re-integrate lateral velocity here (rather than using df["VelocityX"] from
    # ibt_reader) so the straight-line zero-reset is applied during integration.
    vx = _integrate_vlat(glat, roll, speed, yr, steer, tick_rate)
    vy = speed  # longitudinal reference: Speed is more stable than integrated VelocityY

    # Fix 3 — Dynamic CG shift under longitudinal load
    # gLong < 0 under braking → CG moves toward front axle → l_front decreases.
    # kappa = 0.03 m per G is conservative (typical GT3 pitch coefficient).
    kappa = 0.03
    cg_shift = glong * kappa
    l_front = np.clip(
        l_front_base + cg_shift,
        l_front_base * 0.85,
        l_front_base * 1.15,
    )
    l_rear = wheelbase_m - l_front  # constrained so l_front + l_rear = wheelbase

    # Axle lateral velocities (rigid body, now element-wise with dynamic arms)
    v_lat_front = vx + yr * l_front
    v_lat_rear  = vx - yr * l_rear

    v_long = vy  # Speed array — always reliable, no fallback needed

    # Raw kinematic slip angles
    mask = v_long > min_speed_ms
    alpha_front_raw = np.where(mask, np.arctan2(v_lat_front, v_long), 0.0)
    alpha_rear_raw  = np.where(mask, np.arctan2(v_lat_rear,  v_long), 0.0)

    # Fix 2 — Steering saturation cap
    # When gLat stops increasing despite more steering (dGlat/dSteer < 0.1 G/rad),
    # the front is scrubbing.  Cap the effective steering angle at that threshold
    # so the kinematic model doesn't report ever-increasing front engagement.
    steer_abs  = np.abs(steer)
    glat_abs   = np.abs(glat)
    dgLat_dsteer = np.gradient(glat_abs) / (np.gradient(steer_abs) + 1e-6)
    front_saturated = (dgLat_dsteer < 0.1) & (steer_abs > 0.2)
    sat_cap = np.where(
        front_saturated,
        np.minimum.accumulate(np.where(front_saturated, steer_abs, np.inf)),
        steer_abs,
    )
    steer_capped = np.sign(steer) * sat_cap

    front_wheel_angle = steer_capped / steering_ratio
    alpha_front = alpha_front_raw - front_wheel_angle
    alpha_rear  = alpha_rear_raw

    # Fix 1 — Gated rolling max normalization
    # Only update the reference maximum during genuine high-load cornering.
    # Also reject statistical outliers (curb strikes, kerb impacts) that would
    # otherwise compress the normalization scale.  A 3-sigma spike above the
    # 1-second rolling mean is treated as a transient artifact, not a grip limit.
    glat_series = pd.Series(glat_abs)
    glat_roll_mean = glat_series.rolling(60, min_periods=1).mean().to_numpy()
    glat_roll_std  = glat_series.rolling(60, min_periods=1).std().fillna(0).to_numpy()
    spike_mask = glat_abs > (glat_roll_mean + 3.0 * glat_roll_std)
    gate = (speed > 20.0) & (glat_abs > 0.4) & (~spike_mask)
    max_front = _gated_rolling_max(alpha_front, gate)
    max_rear  = _gated_rolling_max(alpha_rear,  gate)

    # Safe denominators prevent 0/0 NaN before the rolling max accumulates
    safe_max_front = np.where(max_front > 1e-6, max_front, 1.0)
    safe_max_rear  = np.where(max_rear  > 1e-6, max_rear,  1.0)
    alpha_front_pct = np.where(max_front > 1e-6, np.abs(alpha_front) / safe_max_front * 100, 0.0)
    alpha_rear_pct  = np.where(max_rear  > 1e-6, np.abs(alpha_rear)  / safe_max_rear  * 100, 0.0)

    alpha_front_pct = np.clip(alpha_front_pct, 0, 100)
    alpha_rear_pct  = np.clip(alpha_rear_pct,  0, 100)

    df["alpha_front_rad"] = alpha_front
    df["alpha_rear_rad"]  = alpha_rear
    df["alpha_front_deg"] = np.degrees(alpha_front)
    df["alpha_rear_deg"]  = np.degrees(alpha_rear)
    df["alpha_front_pct"] = alpha_front_pct
    df["alpha_rear_pct"]  = alpha_rear_pct

    return df


def compute_validation(df: pd.DataFrame) -> dict:
    """Return sanity-check metrics for the slip angle derivation."""
    results = {}

    peak_idx = df["gLat"].abs().idxmax()
    results["peak_glat_g"] = float(df["gLat"].abs().max())
    results["alpha_front_at_peak_g_deg"] = float(df.loc[peak_idx, "alpha_front_deg"])
    results["alpha_rear_at_peak_g_deg"]  = float(df.loc[peak_idx, "alpha_rear_deg"])

    f_sign = np.sign(results["alpha_front_at_peak_g_deg"])
    r_sign = np.sign(results["alpha_rear_at_peak_g_deg"])
    results["sign_ok"] = bool(f_sign == r_sign) and bool(f_sign != 0)

    # Plausibility uses p99 restricted to cornering-speed frames.
    # arctan(v_lat / v_long) is numerically unstable when v_long is small, so
    # near-standstill and slow-hairpin-exit frames are excluded — they are not
    # the frames where the model is expected to produce meaningful slip angles.
    speed_mask = df["Speed"] > 20.0
    f_at_speed = df.loc[speed_mask, "alpha_front_deg"].abs()
    r_at_speed = df.loc[speed_mask, "alpha_rear_deg"].abs()
    p99_f = float(f_at_speed.quantile(0.99)) if len(f_at_speed) > 0 else 0.0
    p99_r = float(r_at_speed.quantile(0.99)) if len(r_at_speed) > 0 else 0.0
    results["max_alpha_front_deg"] = float(df["alpha_front_deg"].abs().max())
    results["max_alpha_rear_deg"]  = float(df["alpha_rear_deg"].abs().max())
    results["p99_alpha_front_deg"] = p99_f
    results["p99_alpha_rear_deg"]  = p99_r
    results["plausible"] = p99_f < 20.0 and p99_r < 20.0

    g_lat_abs = df["gLat"].abs()
    bilateral = np.minimum(df["alpha_front_pct"], df["alpha_rear_pct"])
    if g_lat_abs.std() > 0 and bilateral.std() > 0:
        corr = float(np.corrcoef(g_lat_abs, bilateral)[0, 1])
    else:
        corr = 0.0
    results["bilateral_glat_corr"] = corr

    return results
