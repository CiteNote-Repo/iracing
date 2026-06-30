"""
Three-channel input analysis.

Channel 1 (pull hand) – steering rate in the direction of the corner.
Channel 2 (push hand) – sustained steering angle magnitude.
Channel 3 (feet)      – trail brake and early throttle (longitudinal pre-loading).
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def compute_channels(
    df: pd.DataFrame,
    tick_rate: int = 60,
    brake_threshold: float = 0.05,
    steer_threshold: float = 0.15,  # rad
) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    steer = df["SteeringWheelAngle"].to_numpy(float)
    brake = df["Brake"].to_numpy(float)
    throttle = df["Throttle"].to_numpy(float)
    glat_abs = df["gLat"].abs().to_numpy(float)
    steer_abs = np.abs(steer)

    dt = 1.0 / tick_rate

    # Smooth steering before differentiating
    win = min(15, n if n % 2 == 1 else n - 1)
    win = max(win, 5) | 1
    steer_smooth = savgol_filter(steer, win, 3) if n > win else steer.copy()

    steer_rate = np.gradient(steer_smooth, dt)  # rad/s

    # Channel 1: steering rate weighted by corner direction sign
    # Positive when pulling the inside hand (increasing steer magnitude)
    steer_sign = np.sign(steer_smooth)
    channel1 = steer_rate * steer_sign  # rad/s, positive = pulling into corner

    # Channel 2: steering magnitude sustained through the corner
    channel2 = steer_abs  # rad

    # Channel 2 release rate (negative = push hand releasing)
    channel2_release = -np.gradient(steer_abs, dt)  # positive when steering reduces
    channel2_release = np.maximum(channel2_release, 0.0)

    # Channel 3: trail brake and throttle overlap with lateral load
    trail_brake_active = (brake > brake_threshold) & (steer_abs > steer_threshold)
    throttle_overlap_active = (throttle > brake_threshold) & (steer_abs > steer_threshold)

    channel3_intensity = (brake * glat_abs) + (throttle * glat_abs)

    # Overlap window: brake is releasing AND steering still substantial
    brake_smooth = savgol_filter(brake, win, 3) if n > win else brake.copy()
    brake_rate = np.gradient(brake_smooth, dt)
    brake_releasing = brake_rate < -0.1  # pedal coming off

    # Estimate max steer per corner for the "sustained" criterion
    steer_abs_smooth = savgol_filter(steer_abs, win, 3) if n > win else steer_abs.copy()

    # Overlap active = brake releasing AND steering is ≥70% of recent local max
    # We compute per-sample using a backward rolling max
    steer_local_max = _rolling_backward_max(steer_abs_smooth, window=int(1.5 * tick_rate))
    steer_sustained = steer_abs_smooth > (0.70 * steer_local_max)
    overlap_active = brake_releasing & steer_sustained & (brake > 0.02)

    df["channel1"] = channel1
    df["channel2"] = channel2
    df["channel2_release"] = channel2_release
    df["channel3"] = channel3_intensity
    df["trail_brake"] = trail_brake_active.astype(float)
    df["throttle_overlap"] = throttle_overlap_active.astype(float)
    df["overlap_active"] = overlap_active.astype(float)

    return df


def _rolling_backward_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Max of arr over [i-window, i] at each sample i (vectorised via stride tricks)."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    padded = np.pad(arr, (window, 0), mode="edge")
    # sliding_window_view shape: (n, window+1)
    windows = sliding_window_view(padded, window_shape=window + 1)
    return windows.max(axis=1)[:n]
