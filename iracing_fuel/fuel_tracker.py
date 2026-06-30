import math
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

from logger import log_lap, log_info, log_error

# States
WAITING = "WAITING"
PRACTICE = "PRACTICE"
RACE_INSUFFICIENT_DATA = "RACE_INSUFFICIENT_DATA"
RACE_READY = "RACE_READY"
PITTING = "PITTING"
FINISHED = "FINISHED"


@dataclass
class LapRecord:
    lap_num: int
    fuel_used: float
    lap_time: float
    clean: bool                 # included in fuel average
    is_slow_lap: bool = False   # safety car / anomalously slow — excluded from time average
    exclusion_reason: str = ""


@dataclass
class FuelState:
    state: str = WAITING
    avg_fuel_per_lap: float = 0.0
    avg_lap_time: float = 0.0
    current_fuel: float = 0.0
    fuel_pct: float = 0.0
    tank_capacity: float = 0.0
    session_time_remain: float = 0.0
    session_laps_remain: int = 0
    est_laps_remaining: float = 0.0
    fuel_needed: float = 0.0
    fuel_to_add: float = 0.0
    laps_on_current_tank: float = 0.0
    current_lap: int = 0
    clean_lap_count: int = 0
    last_pit_fuel_set: float = 0.0
    is_timed_race: bool = True


class FuelTracker:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.state = FuelState()

        self._lap_records: List[LapRecord] = []
        self._lap_start_fuel: float = 0.0
        self._lap_start_time: float = 0.0
        self._prev_lap_num: int = 0
        self._prev_on_pit: bool = False
        self._prev_fuel: float = 0.0
        self._lap_dirty: bool = False  # True if IsOnTrack was False at any point this lap

        # Pit detection
        self._pit_entry_callback = None
        self._post_pit_outlap: bool = False  # Next lap after pits is excluded

        # For test mode
        self._test_mode: bool = False

    def set_pit_entry_callback(self, fn):
        self._pit_entry_callback = fn

    def _clean_laps(self) -> List[LapRecord]:
        return [r for r in self._lap_records if r.clean]

    def _rolling_window(self) -> List[LapRecord]:
        window = self.cfg["rolling_window_laps"]
        clean = self._clean_laps()
        return clean[-window:]

    def _calc_weighted_avg_fuel(self) -> float:
        window = self._rolling_window()
        if not window:
            return 0.0
        # weights: older laps get weight 1, most recent gets 3x
        base_weights = [1, 1, 1, 2, 3]
        n = len(window)
        weights = base_weights[-n:]
        total_w = sum(weights)
        return sum(r.fuel_used * w for r, w in zip(window, weights)) / total_w

    def _calc_weighted_avg_time(self, window: List[LapRecord]) -> float:
        # Exclude safety car / anomalously slow laps from time average
        normal = [r for r in window if not r.is_slow_lap and r.lap_time > 0]
        if not normal:
            # Fall back to all laps if every lap was anomalous
            normal = [r for r in window if r.lap_time > 0]
        if not normal:
            return 0.0
        base_weights = [1, 1, 1, 2, 3]
        n = len(normal)
        weights = base_weights[-n:]
        total_w = sum(weights)
        return sum(r.lap_time * w for r, w in zip(normal, weights)) / total_w

    def _fuel_needed_calc(self, avg_fuel: float, avg_time: float, current_fuel: float,
                           session_time_remain: float, session_laps_remain: int,
                           is_timed: bool) -> tuple:
        if is_timed:
            if avg_time <= 0:
                return 0.0, 0.0, 0.0
            est_laps = session_time_remain / avg_time
            laps_to_fuel = est_laps + self.cfg["extra_buffer_laps"]
            fuel_needed = laps_to_fuel * avg_fuel
            fuel_to_add = max(0.0, fuel_needed - current_fuel)
            fuel_to_add *= (1 + self.cfg["safety_margin_pct"] / 100)
            fuel_to_add = math.ceil(fuel_to_add * 2) / 2
            return est_laps, fuel_needed, fuel_to_add
        else:
            laps_left = session_laps_remain
            fuel_needed = laps_left * avg_fuel
            fuel_to_add = max(0.0, fuel_needed - current_fuel)
            fuel_to_add *= (1 + self.cfg["safety_margin_pct"] / 100)
            fuel_to_add = math.ceil(fuel_to_add * 2) / 2
            return float(laps_left), fuel_needed, fuel_to_add

    def update(self, telemetry: dict) -> Optional[float]:
        """
        Called every tick with telemetry dict. Returns fuel amount to command if pit entry
        was detected and enough data exists, otherwise None.
        """
        fuel_level = telemetry.get("FuelLevel", 0.0)
        fuel_pct = telemetry.get("FuelLevelPct", 0.0)
        lap_num = telemetry.get("Lap", 0)
        lap_time = telemetry.get("LapLastLapTime", 0.0)
        session_time_remain = telemetry.get("SessionTimeRemain", 0.0)
        session_laps_remain = telemetry.get("SessionLapsRemain", 9999)
        session_type = telemetry.get("SessionType", "")
        is_on_track = telemetry.get("IsOnTrack", True)
        on_pit_road = telemetry.get("OnPitRoad", False)
        is_race = "race" in session_type.lower()

        with self.lock:
            # Detect tank capacity
            if fuel_pct and fuel_pct > 0.01:
                tank_cap = fuel_level / fuel_pct
            else:
                tank_cap = self.state.tank_capacity or 0.0

            # Detect race type
            is_timed = session_laps_remain > 500 or "time" in session_type.lower()

            # Detect lap change
            lap_changed = (lap_num > self._prev_lap_num and self._prev_lap_num > 0)

            # Track if lap had off-track moments
            if not is_on_track:
                self._lap_dirty = True

            # Mark pit entry (rising edge)
            pit_entry = on_pit_road and not self._prev_on_pit
            fuel_to_command = None

            if lap_changed:
                # Process completed lap
                fuel_used = self._lap_start_fuel - fuel_level
                completed_lap_time = lap_time  # iRacing gives last lap time

                # Determine if this lap should be included in rolling avg
                exclusion = ""
                is_slow_lap = False
                if self._post_pit_outlap:
                    exclusion = "outlap"
                    self._post_pit_outlap = False
                elif self._lap_dirty:
                    exclusion = "off-track"
                elif fuel_used < 0:
                    exclusion = "refueled-mid-lap"
                elif completed_lap_time > 0 and self._lap_records:
                    recent_clean = [r for r in self._lap_records if r.clean and r.lap_time > 0 and not r.is_slow_lap]
                    if recent_clean:
                        avg_t = sum(r.lap_time for r in recent_clean[-5:]) / len(recent_clean[-5:])
                        if completed_lap_time < avg_t * 0.6:
                            exclusion = "too-short"
                        elif completed_lap_time > avg_t * 1.5:
                            # Safety car / red flag: include in fuel avg, exclude from time avg
                            is_slow_lap = True

                is_clean = exclusion == ""
                rec = LapRecord(
                    lap_num=self._prev_lap_num,
                    fuel_used=max(0.0, fuel_used),
                    lap_time=completed_lap_time,
                    clean=is_clean,
                    is_slow_lap=is_slow_lap,
                    exclusion_reason=exclusion if exclusion else ("safety-car" if is_slow_lap else ""),
                )
                self._lap_records.append(rec)

                avg_f = self._calc_weighted_avg_fuel()
                log_lap(self._prev_lap_num, rec.fuel_used, rec.lap_time, avg_f, exclusion)

                # Reset lap tracking
                self._lap_start_fuel = fuel_level
                self._lap_dirty = False

            # First tick of a new session or first lap
            if self._prev_lap_num == 0 and lap_num > 0:
                self._lap_start_fuel = fuel_level
                self._lap_dirty = False

            # Detect if fuel jumped up (refuel during pit) — don't count as consumption
            if fuel_level > self._prev_fuel + 0.5 and self._prev_fuel > 0:
                self._lap_start_fuel = fuel_level
                self._post_pit_outlap = True

            # Pit entry handling
            if pit_entry and is_race:
                avg_fuel = self._calc_weighted_avg_fuel()
                window = self._rolling_window()
                avg_time = self._calc_weighted_avg_time(window)
                clean_count = len(self._clean_laps())

                if clean_count >= self.cfg["min_laps_before_auto"] and avg_fuel > 0:
                    est_laps, fuel_needed, fuel_add = self._fuel_needed_calc(
                        avg_fuel, avg_time, fuel_level,
                        session_time_remain, session_laps_remain, is_timed,
                    )
                    # Cap to tank capacity minus current fuel
                    if tank_cap > 0:
                        max_add = tank_cap - fuel_level
                        fuel_add = min(fuel_add, max_add)
                    fuel_to_command = fuel_add
                    self.state.last_pit_fuel_set = fuel_add

            # Recalculate display state
            avg_fuel = self._calc_weighted_avg_fuel()
            window = self._rolling_window()
            avg_time = self._calc_weighted_avg_time(window)
            clean_count = len(self._clean_laps())

            est_laps, fuel_needed, fuel_to_add = 0.0, 0.0, 0.0
            if avg_fuel > 0 and avg_time > 0:
                est_laps, fuel_needed, fuel_to_add = self._fuel_needed_calc(
                    avg_fuel, avg_time, fuel_level,
                    session_time_remain, session_laps_remain, is_timed,
                )

            laps_on_tank = (fuel_level / avg_fuel) if avg_fuel > 0 else 0.0

            # Determine state
            if not is_race:
                new_state = PRACTICE
            elif on_pit_road:
                new_state = PITTING
            elif clean_count < self.cfg["min_laps_before_auto"]:
                new_state = RACE_INSUFFICIENT_DATA
            else:
                new_state = RACE_READY

            self.state.state = new_state
            self.state.avg_fuel_per_lap = avg_fuel
            self.state.avg_lap_time = avg_time
            self.state.current_fuel = fuel_level
            self.state.fuel_pct = fuel_pct
            self.state.tank_capacity = tank_cap
            self.state.session_time_remain = session_time_remain
            self.state.session_laps_remain = session_laps_remain
            self.state.est_laps_remaining = est_laps
            self.state.fuel_needed = fuel_needed
            self.state.fuel_to_add = fuel_to_add
            self.state.laps_on_current_tank = laps_on_tank
            self.state.current_lap = lap_num
            self.state.clean_lap_count = clean_count
            self.state.is_timed_race = is_timed

            self._prev_lap_num = lap_num
            self._prev_on_pit = on_pit_road
            self._prev_fuel = fuel_level

        return fuel_to_command

    def get_state(self) -> FuelState:
        with self.lock:
            import copy
            return copy.copy(self.state)

    def reset_session(self):
        with self.lock:
            self._lap_records.clear()
            self._lap_start_fuel = 0.0
            self._prev_lap_num = 0
            self._prev_on_pit = False
            self._prev_fuel = 0.0
            self._lap_dirty = False
            self._post_pit_outlap = False
            self.state = FuelState()
        log_info("Session reset")
