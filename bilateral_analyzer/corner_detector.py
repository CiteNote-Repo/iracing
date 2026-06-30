"""
Corner detection and segmentation.

A corner is any contiguous period where |gLat_smooth| > CORNER_THRESHOLD_G
that lasts at least MIN_CORNER_DURATION seconds.

Each corner is labelled with:
  corner_id   – unique int across the session (monotone); -1 = not a corner
  track_corner_num – which track corner (0-indexed), derived from lap position
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


SMOOTH_WINDOW = 31   # samples @ 60 Hz ≈ 0.5 s
SMOOTH_POLY = 3
CLUSTER_TOLERANCE = 0.03  # fraction of lap for "same" track corner


@dataclass
class Corner:
    corner_id: int
    lap: int
    track_corner_num: int
    entry_idx: int
    apex_idx: int
    exit_idx: int
    entry_pct: float
    apex_pct: float
    exit_pct: float
    entry_time: float
    duration: float          # seconds
    peak_glat_g: float
    # populated later by bilateral_scorer
    peak_bilateral: float = 0.0
    mean_bilateral: float = 0.0
    axle_leader: str = "balanced"
    overlap_duration: float = 0.0
    bilateral_during_overlap: float = 0.0
    throttle_app_time: float = float("nan")   # seconds from entry
    bilateral_peak_time: float = float("nan") # seconds from entry


def detect_corners(
    df: pd.DataFrame,
    corner_threshold_g: float = 0.3,
    min_duration_s: float = 0.5,
    tick_rate: int = 60,
) -> tuple[pd.DataFrame, list[Corner], list[float]]:
    """
    Returns:
      df – with added 'corner_id' and 'track_corner_num' columns
      corners – list of Corner objects
      track_corner_positions – list of canonical LapDistPct per unique track corner
    """
    df = df.copy()
    n = len(df)

    glat = df["gLat"].to_numpy(float)
    win = min(SMOOTH_WINDOW, n if n % 2 == 1 else n - 1)
    win = max(win, 5) | 1  # ensure odd
    poly = min(SMOOTH_POLY, win - 1)
    glat_smooth = savgol_filter(glat, win, poly) if n > win else glat.copy()

    in_corner = np.abs(glat_smooth) > corner_threshold_g
    min_samples = int(min_duration_s * tick_rate)

    # Find contiguous runs
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if in_corner[i]:
            j = i
            while j < n and in_corner[j]:
                j += 1
            if j - i >= min_samples:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1

    lap_arr = df["Lap"].to_numpy()
    dist_arr = df["LapDistPct"].to_numpy(float)
    time_arr = df["time"].to_numpy(float)

    corner_id_col = np.full(n, -1, dtype=int)
    track_corner_col = np.full(n, -1, dtype=int)

    corners: list[Corner] = []
    track_positions: list[float] = []  # canonical entry pct per track corner

    for run_start, run_end in runs:
        cid = len(corners)
        peak_g_idx = run_start + int(np.argmax(np.abs(glat_smooth[run_start:run_end + 1])))

        entry_pct = float(dist_arr[run_start])
        apex_pct = float(dist_arr[peak_g_idx])
        exit_pct = float(dist_arr[run_end])
        entry_time = float(time_arr[run_start])
        duration = float(time_arr[run_end] - entry_time)
        lap = int(lap_arr[run_start])

        # Assign to a track corner by matching entry_pct within tolerance
        track_num = _match_track_corner(entry_pct, track_positions)
        if track_num == -1:
            track_num = len(track_positions)
            track_positions.append(entry_pct)

        corner_id_col[run_start:run_end + 1] = cid
        track_corner_col[run_start:run_end + 1] = track_num

        corners.append(Corner(
            corner_id=cid,
            lap=lap,
            track_corner_num=track_num,
            entry_idx=run_start,
            apex_idx=peak_g_idx,
            exit_idx=run_end,
            entry_pct=entry_pct,
            apex_pct=apex_pct,
            exit_pct=exit_pct,
            entry_time=entry_time,
            duration=duration,
            peak_glat_g=float(np.abs(glat_smooth[peak_g_idx])),
        ))

    df["corner_id"] = corner_id_col
    df["track_corner_num"] = track_corner_col
    return df, corners, track_positions


def _match_track_corner(entry_pct: float, positions: list[float]) -> int:
    for i, p in enumerate(positions):
        if abs(entry_pct - p) < CLUSTER_TOLERANCE:
            return i
    # Handle wrap-around at start/finish
    for i, p in enumerate(positions):
        if abs(entry_pct - p + 1.0) < CLUSTER_TOLERANCE or abs(entry_pct - p - 1.0) < CLUSTER_TOLERANCE:
            return i
    return -1


def get_lap_times(df: pd.DataFrame, min_duration_s: float = 10.0) -> dict[int, float]:
    """Return {lap_number: lap_duration_seconds} derived from SessionTime spans.

    Laps shorter than min_duration_s are excluded — this filters the one-frame
    glitch at session start (Lap counter resets to 0) and partial entry laps that
    appear when a recording begins mid-race at arbitrary lap numbers.
    """
    lap_times: dict[int, float] = {}
    for lap in sorted(df["Lap"].unique()):
        lap_df = df[df["Lap"] == lap]
        if len(lap_df) < 10:
            continue
        duration = float(lap_df["SessionTime"].iloc[-1] - lap_df["SessionTime"].iloc[0])
        if duration < min_duration_s:
            continue
        lap_times[int(lap)] = duration
    return lap_times
