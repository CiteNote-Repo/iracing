"""
Synthetic telemetry generator for --demo mode.

Produces four laps representing:
  Lap 1: bilateral   – both axles simultaneous at corners
  Lap 2: oversteer   – rear always leads front
  Lap 3: understeer  – front always leads rear
  Lap 4: sequential  – rear then front, never together
"""

import math
import numpy as np
import pandas as pd

TICK_RATE = 60          # Hz
LAP_DURATION = 90.0     # seconds per lap
NUM_LAPS = 4

# Corner positions (LapDistPct) and duration parameters
CORNERS = [
    {"centre": 0.12, "half_width": 0.04, "peak_g": 0.55},  # tight hairpin
    {"centre": 0.28, "half_width": 0.06, "peak_g": 0.42},  # medium
    {"centre": 0.50, "half_width": 0.03, "peak_g": 0.65},  # fast
    {"centre": 0.71, "half_width": 0.05, "peak_g": 0.48},  # medium
    {"centre": 0.88, "half_width": 0.03, "peak_g": 0.58},  # fast chicane
]


def _corner_profile(dist_pct: float, centre: float, half_width: float) -> float:
    """Gaussian bell centred at corner apex, returns 0-1."""
    sigma = half_width / 2.0
    return math.exp(-0.5 * ((dist_pct - centre) / sigma) ** 2)


def _make_corner_glat(dist_arr: np.ndarray) -> np.ndarray:
    glat = np.zeros(len(dist_arr))
    for corner in CORNERS:
        for i, d in enumerate(dist_arr):
            glat[i] += corner["peak_g"] * _corner_profile(d, corner["centre"], corner["half_width"])
    return glat


def _make_steer(dist_arr: np.ndarray, glat: np.ndarray) -> np.ndarray:
    # Steering proportional to gLat but with some phase lead (entry) and lag (exit)
    steer = np.zeros(len(dist_arr))
    for corner in CORNERS:
        for i, d in enumerate(dist_arr):
            prof = _corner_profile(d, corner["centre"] - 0.01, corner["half_width"])
            steer[i] += 0.4 * prof  # ~0.4 rad peak
    return steer


def _make_brake(dist_arr: np.ndarray) -> np.ndarray:
    brake = np.zeros(len(dist_arr))
    for corner in CORNERS:
        entry = corner["centre"] - corner["half_width"] * 1.8
        apex = corner["centre"]
        for i, d in enumerate(dist_arr):
            if entry <= d <= apex:
                frac = (d - entry) / (apex - entry)
                brake[i] += (1.0 - frac) * 0.7  # tapering trail brake
    return np.clip(brake, 0, 1)


def _make_throttle(dist_arr: np.ndarray) -> np.ndarray:
    throttle = np.ones(len(dist_arr))  # default WOT on straights
    for corner in CORNERS:
        entry = corner["centre"] - corner["half_width"] * 1.5
        exit_ = corner["centre"] + corner["half_width"] * 1.5
        for i, d in enumerate(dist_arr):
            if entry <= d <= exit_:
                frac = (d - corner["centre"]) / (corner["half_width"] * 1.5)
                throttle[i] = max(0, frac)  # rises from 0 at apex to 1 at exit
    return np.clip(throttle, 0, 1)


def _derive_slip_profiles(dist_arr: np.ndarray, glat: np.ndarray, style: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (alpha_front_pct, alpha_rear_pct) for a given driving style.
    The values are 0-100% of their respective peaks.
    """
    n = len(dist_arr)
    alpha_f = np.zeros(n)
    alpha_r = np.zeros(n)

    for corner in CORNERS:
        centre = corner["centre"]
        hw = corner["half_width"]

        for i, d in enumerate(dist_arr):
            g = _corner_profile(d, centre, hw)

            if style == "bilateral":
                # Both peak together, nearly equal
                alpha_f[i] += g * 88
                alpha_r[i] += g * 85

            elif style == "oversteer":
                # Rear spikes first and higher; front engages late
                rear_peak = _corner_profile(d, centre - hw * 0.3, hw * 0.7)
                front_peak = _corner_profile(d, centre + hw * 0.2, hw * 0.8)
                alpha_r[i] += rear_peak * 90
                alpha_f[i] += front_peak * 45

            elif style == "understeer":
                # Front builds early and stays high; rear barely engages
                front_peak = _corner_profile(d, centre - hw * 0.2, hw * 0.8)
                rear_peak = _corner_profile(d, centre + hw * 0.3, hw * 0.7)
                alpha_f[i] += front_peak * 90
                alpha_r[i] += rear_peak * 40

            elif style == "sequential":
                # Rear peaks at entry; front peaks at exit; never together
                rear_peak = _corner_profile(d, centre - hw * 0.6, hw * 0.55)
                front_peak = _corner_profile(d, centre + hw * 0.5, hw * 0.55)
                alpha_r[i] += rear_peak * 85
                alpha_f[i] += front_peak * 82

    alpha_f = np.clip(alpha_f, 0, 100)
    alpha_r = np.clip(alpha_r, 0, 100)
    return alpha_f, alpha_r


def generate_demo_data() -> pd.DataFrame:
    frames_per_lap = int(TICK_RATE * LAP_DURATION)
    total_frames = frames_per_lap * NUM_LAPS
    styles = ["bilateral", "oversteer", "understeer", "sequential"]

    all_laps: list[pd.DataFrame] = []

    for lap_idx, style in enumerate(styles):
        lap_num = lap_idx + 1
        t = np.linspace(0, LAP_DURATION, frames_per_lap, endpoint=False)
        t_abs = t + lap_idx * LAP_DURATION
        dist = t / LAP_DURATION   # 0 to 1

        glat = _make_corner_glat(dist)
        steer = _make_steer(dist, glat)
        brake = _make_brake(dist)
        throttle = _make_throttle(dist)
        speed = 40.0 + (1 - glat / glat.max()) * 40  # slower in corners
        yaw_rate = glat * 10  # rough: YawRate ∝ gLat

        # Derive slip angles
        alpha_f_pct, alpha_r_pct = _derive_slip_profiles(dist, glat, style)

        # Build VelocityX/Y consistent with slip angles (rough physical scaling)
        # alpha = atan(VX / VY) → VX ≈ VY * alpha_rad
        alpha_f_rad = np.radians(alpha_f_pct / 100 * 8)  # map 0-100% to 0-8°
        alpha_r_rad = np.radians(alpha_r_pct / 100 * 8)
        vel_long = speed.copy()
        vel_lat = alpha_r_rad * vel_long  # rear axle at CG approximation

        add_noise = lambda arr, scale: arr + np.random.randn(len(arr)) * scale

        lap_df = pd.DataFrame({
            "SessionTime": t_abs,
            "time": t_abs,
            "Speed": add_noise(speed, 0.3),
            "VelocityX": add_noise(vel_lat, 0.05),
            "VelocityY": add_noise(vel_long, 0.5),
            "YawRate": add_noise(yaw_rate, 0.1),
            "SteeringWheelAngle": add_noise(steer, 0.01),
            "gLat": add_noise(glat, 0.02),
            "gLong": np.zeros(frames_per_lap),
            "Throttle": np.clip(add_noise(throttle, 0.02), 0, 1),
            "Brake": np.clip(add_noise(brake, 0.02), 0, 1),
            "LapDistPct": dist,
            "Lap": lap_num,
            "LapLastLapTime": LAP_DURATION if lap_num > 1 else 0.0,
            # Pre-computed slip angle percentages (bypass physics derivation for demo)
            "_alpha_front_pct_demo": alpha_f_pct,
            "_alpha_rear_pct_demo": alpha_r_pct,
        })
        all_laps.append(lap_df)

    df = pd.concat(all_laps, ignore_index=True)
    return df


def apply_demo_slip_angles(df: pd.DataFrame) -> pd.DataFrame:
    """Use the pre-computed demo slip angles instead of deriving from physics."""
    df = df.copy()
    if "_alpha_front_pct_demo" in df.columns:
        df["alpha_front_pct"] = df["_alpha_front_pct_demo"]
        df["alpha_rear_pct"] = df["_alpha_rear_pct_demo"]
        # Approximate degrees (0-8° range)
        df["alpha_front_deg"] = df["alpha_front_pct"] / 100 * 8
        df["alpha_rear_deg"] = df["alpha_rear_pct"] / 100 * 8
        df["alpha_front_rad"] = np.radians(df["alpha_front_deg"])
        df["alpha_rear_rad"] = np.radians(df["alpha_rear_deg"])
        df.drop(columns=["_alpha_front_pct_demo", "_alpha_rear_pct_demo"], inplace=True)
    return df
