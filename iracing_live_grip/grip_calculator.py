from dataclasses import dataclass
from typing import Optional

import numpy as np

QUIET = "QUIET"
PEAK_GRIP_HISS = "PEAK_GRIP_HISS"
REAR_SLIDE = "REAR_SLIDE"
FRONT_LOCK = "FRONT_LOCK"
OVERSTEER_HOWL = "OVERSTEER_HOWL"

TYRE_RADIUS = 0.33  # metres, GT3 approximate (rad/s → m/s conversion)
_G = 9.80665

_scrub_session_max: float = 0.0


@dataclass
class GripData:
    total_util: float = 0.0
    overall_state: str = QUIET
    scrub_proximity_pct: float = 0.0   # 100=front efficient, low=scrubbing; 0 when no data
    yaw_deviation_pct: float = 0.0     # +oversteer, -understeer; 0 when gates not met
    rear_slip_raw: float = 0.0         # kept for compatibility; always 0
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
    wheelbase_m: float = 2.63,
) -> GripData:
    global _scrub_session_max

    glat_abs = abs(gLat)
    total_util = min(glat_abs / peak_glat_g * 100.0, 130.0)

    # Scrub proximity: lateral G per radian of road-wheel angle relative to
    # the session rolling maximum.  100% = front at peak efficiency; lower
    # values mean the front is returning less G per degree of steering —
    # i.e. it is scrubbing past its peak.  Gated to cornering conditions where
    # the signal is meaningful; decays slowly so brief peaks don't permanently
    # anchor the reference.
    scrub_proximity_pct = 0.0
    road_wheel_angle = abs(steering_angle) / steering_ratio
    if road_wheel_angle > 0.05 and total_util > 40.0 and speed > 20.0:
        current_efficiency = glat_abs / road_wheel_angle
        _scrub_session_max = max(current_efficiency, _scrub_session_max * 0.9995)
        if _scrub_session_max > 0:
            scrub_proximity_pct = min(current_efficiency / _scrub_session_max * 100.0, 100.0)

    # Yaw-rate deviation: compare actual yaw rate to the neutral-steer prediction.
    # Positive = rear rotating faster than steering predicts (oversteer).
    # Negative = rear rotating slower (understeer / front scrub).
    yaw_deviation_pct = 0.0
    if speed > 10.0 and abs(steering_angle) > 0.05:
        predicted_yaw = (speed * abs(steering_angle)) / (steering_ratio * wheelbase_m)
        raw_deviation = abs(yaw_rate) - predicted_yaw
        yaw_deviation_norm = raw_deviation * wheelbase_m / max(speed, 1.0)
        yaw_deviation_pct = float(np.clip(yaw_deviation_norm * 200, -100, 100))

    # Map yaw deviation to acoustic state: treat oversteer > 15% like a rear slide
    yaw_as_slip = (yaw_deviation_pct / 100.0) if yaw_deviation_pct > 15.0 else None
    overall_state = classify_acoustic_state(total_util, rear_slip=yaw_as_slip)

    return GripData(
        total_util=total_util,
        overall_state=overall_state,
        scrub_proximity_pct=scrub_proximity_pct,
        yaw_deviation_pct=yaw_deviation_pct,
        rear_slip_raw=0.0,
        is_on_track=is_on_track,
        speed_mps=speed,
        connected=True,
    )
