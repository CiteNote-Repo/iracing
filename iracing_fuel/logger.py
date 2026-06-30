import os
import threading
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "iracing_fuel_log.txt")

_lock = threading.Lock()


def _write(line: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with _lock:
        with open(LOG_PATH, "a") as f:
            f.write(f"[{ts}] {line}\n")
    print(f"[{ts}] {line}")


def log_lap(lap_num: int, fuel_used: float, lap_time: float, avg_fuel: float, excluded: str = "") -> None:
    if excluded:
        _write(f"LAP {lap_num:3d} | fuel={fuel_used:.3f}L | time={lap_time:.2f}s | avg={avg_fuel:.3f}L/lap | EXCLUDED: {excluded}")
    else:
        _write(f"LAP {lap_num:3d} | fuel={fuel_used:.3f}L | time={lap_time:.2f}s | avg={avg_fuel:.3f}L/lap")


def log_pit(fuel_on_entry: float, fuel_commanded: float, session_time_remain: float) -> None:
    mins = int(session_time_remain // 60)
    secs = int(session_time_remain % 60)
    _write(f"PIT    | entry_fuel={fuel_on_entry:.2f}L | commanded={fuel_commanded:.1f}L | time_remain={mins}:{secs:02d}")


def log_info(msg: str) -> None:
    _write(f"INFO   | {msg}")


def log_error(msg: str) -> None:
    _write(f"ERROR  | {msg}")
