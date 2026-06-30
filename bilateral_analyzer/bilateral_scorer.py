"""
Bilateral balance scoring and per-corner / per-lap metric extraction.

bilateral_score = min(alpha_front_pct, alpha_rear_pct)   → 0-100
axle_imbalance  = alpha_front_pct - alpha_rear_pct       → + = front-led
"""

import math
import numpy as np
import pandas as pd

from corner_detector import Corner
from tyre_energy_tracker import QUIET, PEAK_GRIP_HISS, REAR_SLIDE, FRONT_LOCK, OVERSTEER_HOWL, STATE_NAMES


# Calibrate once from the fastest clean laps for the car+track combo.
# Update via bilateral_config.json "peak_glat_g" key.
PEAK_GLAT_DEFAULT = 2.30  # G — Ferrari 296 GT3 at Bathurst on warm tyres


def compute_bilateral(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Gracefully degrade when slip angles haven't been computed yet
    if "alpha_front_pct" in df.columns and "alpha_rear_pct" in df.columns:
        df["bilateral_score"] = np.minimum(df["alpha_front_pct"], df["alpha_rear_pct"])
        df["axle_imbalance"]  = df["alpha_front_pct"] - df["alpha_rear_pct"]
    else:
        df["bilateral_score"] = 0.0
        df["axle_imbalance"]  = 0.0

    # Fix 4 — Understeer gradient metric
    # Speed × YawRate = centripetal acceleration a neutral car would need (m/s²).
    # LatAccel is what the chassis actually experiences.
    # Difference / speed² is speed-independent and discriminates axle balance:
    #   positive → understeer (lateral G below what yaw rate predicts)
    #   negative → oversteer  (lateral G above what yaw rate predicts)
    #   near zero → neutral / bilateral engagement
    expected_lat = df["Speed"] * df["YawRate"]          # m/s²
    actual_lat   = df["LatAccel"]                        # m/s²
    speed_sq     = np.maximum(df["Speed"] ** 2, 1.0)
    ug = (actual_lat - expected_lat) / speed_sq

    df["understeer_gradient"] = ug

    # Normalize against cornering samples only — non-cornering frames have tiny
    # speed_sq (clamped at 1.0) so standstill/pit-lane spikes would otherwise
    # dominate the 99th-percentile scale and compress the cornering signal to near-zero.
    corn_mask = (df["Speed"] > 20) & (df["gLat"].abs() > 0.4)
    ug_corn = ug[corn_mask] if corn_mask.any() else ug
    max_ug = float(ug_corn.abs().quantile(0.99))
    if max_ug > 1e-6:
        df["understeer_gradient_pct"] = (ug / max_ug * 100).clip(-100, 100)
    else:
        df["understeer_gradient_pct"] = 0.0

    return df


def compute_absolute_utilization(
    df: pd.DataFrame,
    peak_glat_g: float = PEAK_GLAT_DEFAULT,
) -> pd.DataFrame:
    """
    Compute lateral tyre utilization as a fraction of known peak G.

    Unlike the kinematic slip angle derivation, this uses gLat directly —
    a quantity iRacing computes correctly at its internal physics rate and
    exposes reliably in telemetry.  It does not depend on the vehicle model
    (wheelbase, steering ratio, CG position) or session-relative normalization.

    Columns added:
      lateral_utilization_pct — |gLat| / peak_glat as 0-110%
      axle_balance_pct        — (gLat - Speed*YawRate/g) / peak_glat as ±100%
                                positive = understeer (front working harder than
                                           yaw rate would predict)
                                negative = oversteer  (rotation exceeds what
                                           steering contribution implies)
                                near zero = balanced bilateral engagement
    """
    _G = 9.80665
    glat_g  = df["gLat"].abs()
    yaw_g   = (df["Speed"] * df["YawRate"].abs()) / _G   # centripetal G from rotation
    steer_g = glat_g - yaw_g                              # residual G attributed to steering

    df = df.copy()
    df["lateral_utilization_pct"] = (glat_g / peak_glat_g * 100).clip(0, 110)
    df["axle_balance_pct"]        = (steer_g / peak_glat_g * 100).clip(-100, 100)
    return df


def score_corners(
    df: pd.DataFrame,
    corners: list[Corner],
    tick_rate: int = 60,
    bilateral_threshold: float = 70.0,
) -> list[Corner]:
    """Populate per-corner stats on each Corner object (mutates in place)."""
    steer_abs = df["SteeringWheelAngle"].abs().to_numpy(float)
    time_arr = df["time"].to_numpy(float)
    bilateral = df["bilateral_score"].to_numpy(float)
    alpha_front = df["alpha_front_pct"].to_numpy(float)
    alpha_rear = df["alpha_rear_pct"].to_numpy(float)
    throttle = df["Throttle"].to_numpy(float)
    overlap = df["overlap_active"].to_numpy(float)

    for c in corners:
        s, e = c.entry_idx, c.exit_idx
        if s >= e:
            continue

        seg_b = bilateral[s:e + 1]
        seg_f = alpha_front[s:e + 1]
        seg_r = alpha_rear[s:e + 1]
        seg_t = throttle[s:e + 1]
        seg_ov = overlap[s:e + 1]
        seg_time = time_arr[s:e + 1]

        c.peak_bilateral = float(seg_b.max())
        c.mean_bilateral = float(seg_b.mean())

        peak_rel_idx = int(seg_b.argmax())
        c.bilateral_peak_time = float(seg_time[peak_rel_idx] - c.entry_time)

        # Axle leader: average imbalance through mid-corner
        mid_s = len(seg_b) // 4
        mid_e = 3 * len(seg_b) // 4
        mean_f = float(seg_f[mid_s:mid_e].mean())
        mean_r = float(seg_r[mid_s:mid_e].mean())
        if mean_f - mean_r > 15:
            c.axle_leader = "front"
        elif mean_r - mean_f > 15:
            c.axle_leader = "rear"
        else:
            c.axle_leader = "balanced"

        # Overlap metrics
        ov_samples = int(seg_ov.sum())
        c.overlap_duration = ov_samples / tick_rate
        if ov_samples > 0:
            c.bilateral_during_overlap = float(seg_b[seg_ov.astype(bool)].mean())
        else:
            c.bilateral_during_overlap = 0.0

        # Throttle application time (first sample where throttle > 20%)
        throttle_on = np.where(seg_t > 0.20)[0]
        if len(throttle_on) > 0:
            c.throttle_app_time = float(seg_time[throttle_on[0]] - c.entry_time)

    return corners


def score_laps(
    df: pd.DataFrame,
    corners: list[Corner],
    lap_times: dict[int, float],
    bilateral_threshold: float = 70.0,
) -> list[dict]:
    """Return per-lap summary dicts."""
    laps_in_data = df["Lap"].unique()
    lap_summaries: list[dict] = []

    for lap_num in sorted(laps_in_data):
        lap_corners = [c for c in corners if c.lap == int(lap_num)]
        if not lap_corners:
            continue

        lap_mask = df["Lap"] == lap_num
        lap_bilateral = df.loc[lap_mask, "bilateral_score"]

        good_corners = sum(
            1 for c in lap_corners if c.peak_bilateral >= bilateral_threshold
        )
        rear_led  = sum(1 for c in lap_corners if c.axle_leader == "rear")
        front_led = sum(1 for c in lap_corners if c.axle_leader == "front")

        # Overlap quality: 0-10 based on mean bilateral-during-overlap
        ov_scores = [c.bilateral_during_overlap for c in lap_corners if c.overlap_duration > 0.05]
        overlap_quality = (np.mean(ov_scores) / 10.0) if ov_scores else 0.0
        overlap_quality = round(min(10.0, overlap_quality), 1)

        # Mean lateral utilization during cornering (absolute metric, car-independent)
        corn_mask = lap_mask & (df["gLat"].abs() > 0.3)
        if "lateral_utilization_pct" in df.columns and corn_mask.any():
            mean_lat_util = float(df.loc[corn_mask, "lateral_utilization_pct"].mean())
        else:
            mean_lat_util = float("nan")

        # Acoustic state breakdown
        tick_rate = df.attrs.get("tick_rate", 60)
        if "acoustic_state" in df.columns:
            lap_state = df.loc[lap_mask, "acoustic_state"].to_numpy(int)
            acoustic_secs = {
                f"{STATE_NAMES[s].lower().replace(' ', '_')}_s": float((lap_state == s).sum()) / tick_rate
                for s in (QUIET, PEAK_GRIP_HISS, REAR_SLIDE, FRONT_LOCK, OVERSTEER_HOWL)
            }
        else:
            acoustic_secs = {
                f"{STATE_NAMES[s].lower().replace(' ', '_')}_s": float("nan")
                for s in (QUIET, PEAK_GRIP_HISS, REAR_SLIDE, FRONT_LOCK, OVERSTEER_HOWL)
            }

        peak_rear_slip = (
            float(df.loc[lap_mask, "rear_slip"].max())
            if "rear_slip" in df.columns
            else float("nan")
        )

        lap_summaries.append({
            "lap": int(lap_num),
            "lap_time": lap_times.get(int(lap_num), float("nan")),
            "mean_bilateral": float(lap_bilateral.mean()),
            "peak_bilateral": float(lap_bilateral.max()),
            "good_corners": good_corners,
            "total_corners": len(lap_corners),
            "rear_led_corners": rear_led,
            "front_led_corners": front_led,
            "overlap_quality": overlap_quality,
            "mean_lateral_utilization": mean_lat_util,
            **acoustic_secs,
            "peak_rear_slip": peak_rear_slip,
        })

    return lap_summaries


def build_track_xy(df: pd.DataFrame, tick_rate: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Integrate speed+yaw to produce approximate 2D track coordinates."""
    dt = 1.0 / tick_rate
    yr = df["YawRate"].to_numpy(float)
    speed = df["Speed"].to_numpy(float)

    heading = np.cumsum(yr * dt)
    x = np.cumsum(speed * np.cos(heading) * dt)
    y = np.cumsum(speed * np.sin(heading) * dt)
    return x, y
