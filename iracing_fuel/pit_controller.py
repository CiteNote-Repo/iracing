import threading
import time

from logger import log_pit, log_info, log_error

try:
    import irsdk
    IRSDK_AVAILABLE = True
except ImportError:
    IRSDK_AVAILABLE = False


class PitController:
    def __init__(self, ir, fuel_tracker, overlay_ref=None):
        self._ir = ir
        self._tracker = fuel_tracker
        self._overlay = overlay_ref
        self._last_pit_time: float = 0.0
        self._cooldown: float = 30.0  # seconds between auto-commands (prevent double-fire)

    def set_overlay(self, overlay):
        self._overlay = overlay

    def handle_pit_entry(self, fuel_to_add: float, session_time_remain: float) -> None:
        now = time.time()
        if now - self._last_pit_time < self._cooldown:
            log_info(f"Pit command skipped — cooldown ({self._cooldown:.0f}s not elapsed)")
            return

        if fuel_to_add <= 0:
            log_info("Pit entry: no fuel needed")
            return

        current_fuel = self._tracker.get_state().current_fuel
        log_pit(current_fuel, fuel_to_add, session_time_remain)

        if IRSDK_AVAILABLE and self._ir is not None:
            try:
                self._ir.pit_command(irsdk.PitCommandMode.fuel, int(fuel_to_add))
                log_info(f"AUTO-SET: {fuel_to_add}L fuel commanded via pit_command")
            except Exception as e:
                log_error(f"pit_command failed: {e}")
        else:
            log_info(f"[TEST MODE] Would command {fuel_to_add}L fuel")

        self._last_pit_time = now

        if self._overlay:
            self._overlay.flash_pit_set(fuel_to_add)
