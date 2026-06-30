from dataclasses import dataclass
from typing import Optional

QUIET = "QUIET"
PEAK_GRIP_HISS = "PEAK_GRIP_HISS"
REAR_SLIDE = "REAR_SLIDE"
FRONT_LOCK = "FRONT_LOCK"
OVERSTEER_HOWL = "OVERSTEER_HOWL"

TYRE_RADIUS = 0.33  # metres, GT3 approximate (rad/s → m/s conversion)
_G = 9.80665


@dataclass
class GripData:
    front_util: float = 0.0
    rear_util: float = 0.0
    total_util: float = 0.0
    front_state: str = QUIET
    rear_state: str = QUIET
    overall_state: str = QUIET
    is_on_track: bool = False
    speed_mps: float = 0.0
    connected: bool = False


def compute_live_utilization(
    gLat: float, speed: float, yaw_rate: float, peak_glat_g: float = 2.30
) -> tuple:
    """Returns (front_util_pct, rear_util_pct, total_util_pct), each 0-130."""
    glat_abs = abs(gLat)
    total_util = min(glat_abs / peak_glat_g * 100.0, 130.0)

    # Rotation-driven G (rear axle signature)
    yaw_g = (speed * abs(yaw_rate)) / _G
    # Steering-driven residual G (front axle signature)
    steer_g = max(glat_abs - yaw_g, 0.0)

    rear_util = min((yaw_g / peak_glat_g) * 100.0, 130.0)
    front_util = min((steer_g / peak_glat_g) * 100.0, 130.0)

    return front_util, rear_util, total_util


def classify_acoustic_state(
    util_pct: float,
    rear_slip: Optional[float] = None,
    front_slip: Optional[float] = None,
) -> str:
    if util_pct < 75.0:
        return QUIET
    elif util_pct < 90.0:
        return PEAK_GRIP_HISS
    elif rear_slip is not None and rear_slip > 0.05:
        return REAR_SLIDE
    elif front_slip is not None and front_slip < -0.05:
        return FRONT_LOCK
    elif util_pct > 100.0:
        return OVERSTEER_HOWL
    else:
        return PEAK_GRIP_HISS


def compute_grip(
    gLat: float,
    speed: float,
    yaw_rate: float,
    is_on_track: bool,
    peak_glat_g: float,
    rear_surface_mps: Optional[float] = None,
    front_surface_mps: Optional[float] = None,
) -> GripData:
    front_util, rear_util, total_util = compute_live_utilization(
        gLat, speed, yaw_rate, peak_glat_g
    )

    rear_slip: Optional[float] = None
    front_slip: Optional[float] = None
    if rear_surface_mps is not None and front_surface_mps is not None and speed > 1.0:
        safe_spd = max(speed, 1.0)
        rear_slip = (rear_surface_mps - speed) / safe_spd
        front_slip = (front_surface_mps - speed) / safe_spd

    # Per-axle: each bar only sees its own slip channel
    front_state = classify_acoustic_state(front_util, front_slip=front_slip)
    rear_state = classify_acoustic_state(rear_util, rear_slip=rear_slip)
    overall_state = classify_acoustic_state(
        total_util, rear_slip=rear_slip, front_slip=front_slip
    )

    return GripData(
        front_util=front_util,
        rear_util=rear_util,
        total_util=total_util,
        front_state=front_state,
        rear_state=rear_state,
        overall_state=overall_state,
        is_on_track=is_on_track,
        speed_mps=speed,
        connected=True,
    )
