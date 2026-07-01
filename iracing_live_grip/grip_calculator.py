from dataclasses import dataclass
from typing import Optional

QUIET = "QUIET"
PEAK_GRIP_HISS = "PEAK_GRIP_HISS"
REAR_SLIDE = "REAR_SLIDE"
FRONT_LOCK = "FRONT_LOCK"
OVERSTEER_HOWL = "OVERSTEER_HOWL"

TYRE_RADIUS = 0.33  # metres, GT3 approximate (rad/s → m/s conversion)
_G = 9.80665

_steer_eff_session_max: float = 0.0


@dataclass
class GripData:
    total_util: float = 0.0
    overall_state: str = QUIET
    steer_efficiency_pct: float = 0.0  # 0-100% of session max; 0 when no data
    rear_slip_raw: float = 0.0         # raw slip fraction; display when abs > 0.03
    is_on_track: bool = False
    speed_mps: float = 0.0
    connected: bool = False


def classify_acoustic_state(
    util_pct: float,
    rear_slip: Optional[float] = None,
) -> str:
    if util_pct < 75.0:
        return QUIET
    elif util_pct < 90.0:
        return PEAK_GRIP_HISS
    elif rear_slip is not None and rear_slip > 0.05:
        return REAR_SLIDE
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
    steering_angle: float = 0.0,
    steering_ratio: float = 13.0,
    rear_surface_mps: Optional[float] = None,
    front_surface_mps: Optional[float] = None,
) -> GripData:
    global _steer_eff_session_max

    glat_abs = abs(gLat)
    total_util = min(glat_abs / peak_glat_g * 100.0, 130.0)

    # Steering efficiency: lateral G produced per radian of road-wheel angle.
    # Only sampled when there's meaningful steering input, speed, and load —
    # below those gates the ratio is dominated by noise.
    steer_efficiency_pct = 0.0
    steer_abs = abs(steering_angle)
    if steer_abs > 0.05 and total_util > 50.0 and speed > 20.0:
        road_wheel_angle = steer_abs / steering_ratio
        raw_eff = glat_abs / road_wheel_angle
        _steer_eff_session_max = max(_steer_eff_session_max, raw_eff)
        if _steer_eff_session_max > 0:
            steer_efficiency_pct = min(raw_eff / _steer_eff_session_max * 100.0, 100.0)

    # Rear slip from wheel speed vs GPS speed
    rear_slip_raw = 0.0
    rear_slip: Optional[float] = None
    if rear_surface_mps is not None and speed > 1.0:
        rear_slip = (rear_surface_mps - speed) / max(speed, 1.0)
        rear_slip_raw = rear_slip

    overall_state = classify_acoustic_state(total_util, rear_slip=rear_slip)

    return GripData(
        total_util=total_util,
        overall_state=overall_state,
        steer_efficiency_pct=steer_efficiency_pct,
        rear_slip_raw=rear_slip_raw,
        is_on_track=is_on_track,
        speed_mps=speed,
        connected=True,
    )
