import copy
import threading
import time
from typing import Callable, Optional

from grip_calculator import GripData, compute_grip, TYRE_RADIUS


class LiveTelemetry:
    def __init__(self, peak_glat_g: float):
        self._peak_glat_g = peak_glat_g
        self._grip_data = GripData()
        self._lock = threading.Lock()
        self._wheel_format: Optional[str] = None  # "rads", "mps", or "none"
        self._running = False

    def get_grip(self) -> GripData:
        with self._lock:
            return copy.copy(self._grip_data)

    def _set_grip(self, data: GripData) -> None:
        """For test mode direct injection."""
        with self._lock:
            self._grip_data = data

    def _detect_wheel_format(self, ir) -> Optional[str]:
        """Probe which wheel-speed naming convention this iRacing build uses."""
        try:
            vals = [
                ir["WheelRRSpeed"], ir["WheelLRSpeed"],
                ir["WheelRFSpeed"], ir["WheelLFSpeed"],
            ]
            if any(v is not None for v in vals):
                if max(abs(v or 0.0) for v in vals) > 0.1:
                    return "rads"
        except Exception:
            pass
        try:
            vals = [ir["RRspeed"], ir["LRspeed"], ir["RFspeed"], ir["LFspeed"]]
            if any(v is not None for v in vals):
                if max(abs(v or 0.0) for v in vals) > 0.1:
                    return "mps"
        except Exception:
            pass
        return None

    def _get_wheel_surfaces(self, ir) -> tuple:
        """Returns (rear_surface_mps, front_surface_mps) or (None, None)."""
        fmt = self._wheel_format
        if fmt is None or fmt == "none":
            return None, None
        try:
            if fmt == "rads":
                rr = float(ir["WheelRRSpeed"] or 0.0)
                lr = float(ir["WheelLRSpeed"] or 0.0)
                rf = float(ir["WheelRFSpeed"] or 0.0)
                lf = float(ir["WheelLFSpeed"] or 0.0)
                return (rr + lr) / 2.0 * TYRE_RADIUS, (rf + lf) / 2.0 * TYRE_RADIUS
            else:
                rr = float(ir["RRspeed"] or 0.0)
                lr = float(ir["LRspeed"] or 0.0)
                rf = float(ir["RFspeed"] or 0.0)
                lf = float(ir["LFspeed"] or 0.0)
                return (rr + lr) / 2.0, (rf + lf) / 2.0
        except Exception:
            return None, None

    def run(self, ir, on_update: Optional[Callable[[GripData], None]] = None) -> None:
        """Live telemetry loop — call in a daemon thread."""
        self._running = True
        connected = False

        while self._running:
            try:
                if not ir.is_initialized or not ir.is_connected:
                    if connected:
                        connected = False
                        self._wheel_format = None
                        with self._lock:
                            self._grip_data = GripData()
                    ir.startup()
                    if ir.is_initialized and ir.is_connected:
                        connected = True
                        print("Connected to iRacing")
                    else:
                        time.sleep(2.0)
                        continue

                ir.freeze_var_buffer_latest()

                is_on_track = bool(ir["IsOnTrack"])
                speed = float(ir["Speed"] or 0.0)
                gLat = float(ir["gLat"] or 0.0)
                yaw_rate = float(ir["YawRate"] or 0.0)
                steering_angle = float(ir["SteeringWheelAngle"] or 0.0)

                # Lazy wheel speed format detection — probe once car is moving
                if self._wheel_format is None and speed > 5.0:
                    self._wheel_format = self._detect_wheel_format(ir) or "none"
                    if self._wheel_format != "none":
                        print(f"Wheel speed format: {self._wheel_format}")

                rear_mps, front_mps = self._get_wheel_surfaces(ir)

                data = compute_grip(
                    gLat=gLat,
                    speed=speed,
                    yaw_rate=yaw_rate,
                    is_on_track=is_on_track,
                    peak_glat_g=self._peak_glat_g,
                    steering_angle=steering_angle,
                    rear_surface_mps=rear_mps,
                    front_surface_mps=front_mps,
                )

                with self._lock:
                    self._grip_data = data

                if on_update:
                    on_update(data)

            except Exception:
                pass

            time.sleep(1.0 / 60.0)

    def stop(self) -> None:
        self._running = False
