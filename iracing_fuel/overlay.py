import threading
import time
import tkinter as tk
from typing import Callable, Optional

from fuel_tracker import (
    WAITING, PRACTICE, RACE_INSUFFICIENT_DATA, RACE_READY, PITTING, FINISHED, FuelState
)

COLOR_GREEN = "#00FF00"
COLOR_YELLOW = "#FFFF00"
COLOR_RED = "#FF3333"
COLOR_WHITE = "#FFFFFF"
COLOR_DIM = "#888888"
COLOR_FLASH = "#00FF88"
BG = "black"

FONT_TITLE = ("Consolas", 10, "bold")
FONT_BODY = ("Consolas", 9)
FONT_STATUS = ("Consolas", 9, "italic")


def _fmt_time(seconds: float) -> str:
    if seconds <= 0:
        return "--:--"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


class FuelOverlay:
    def __init__(self, cfg: dict, get_state_fn: Callable[[], FuelState], on_drag_end: Optional[Callable] = None):
        self._cfg = cfg
        self._get_state = get_state_fn
        self._on_drag_end = on_drag_end

        self._flash_msg: Optional[str] = None
        self._flash_until: float = 0.0
        self._flash_lock = threading.Lock()

        self._root: Optional[tk.Tk] = None
        self._running = False

        # Drag state
        self._drag_x = 0
        self._drag_y = 0

    def flash_pit_set(self, fuel_amount: float) -> None:
        with self._flash_lock:
            self._flash_msg = f"FUEL SET: {fuel_amount:.1f}L"
            self._flash_until = time.time() + 3.0

    def _get_flash(self) -> Optional[str]:
        with self._flash_lock:
            if self._flash_msg and time.time() < self._flash_until:
                return self._flash_msg
            self._flash_msg = None
            return None

    def _fuel_color(self, laps_on_tank: float) -> str:
        if laps_on_tank > 3:
            return COLOR_GREEN
        elif laps_on_tank > 1:
            return COLOR_YELLOW
        else:
            return COLOR_RED

    def _build_ui(self) -> None:
        root = self._root
        pos = self._cfg["overlay_position"]
        alpha = self._cfg.get("overlay_alpha", 0.85)

        root.geometry(f"+{pos['x']}+{pos['y']}")
        root.attributes("-topmost", True)
        root.attributes("-alpha", alpha)
        root.overrideredirect(True)
        root.configure(bg=BG)

        # Windows transparent background trick (no-op on macOS but harmless)
        try:
            root.wm_attributes("-transparentcolor", BG)
        except Exception:
            pass

        # Drag bindings
        root.bind("<Button-1>", self._on_drag_start)
        root.bind("<B1-Motion>", self._on_drag_motion)
        root.bind("<ButtonRelease-1>", self._on_drag_release)

        # Frame with border
        self._frame = tk.Frame(root, bg=BG, bd=1, relief="solid", highlightbackground="#444444",
                               highlightthickness=1)
        self._frame.pack(padx=2, pady=2)
        self._frame.bind("<Button-1>", self._on_drag_start)
        self._frame.bind("<B1-Motion>", self._on_drag_motion)
        self._frame.bind("<ButtonRelease-1>", self._on_drag_release)

        self._labels = {}
        rows = [
            ("title", "  FUEL CALC  ", FONT_TITLE, COLOR_WHITE),
            ("avg_lap", "  Avg/lap: --", FONT_BODY, COLOR_WHITE),
            ("remaining", "  Remaining: --", FONT_BODY, COLOR_WHITE),
            ("est_laps", "  Est laps: --", FONT_BODY, COLOR_WHITE),
            ("fuel_need", "  Fuel need: --", FONT_BODY, COLOR_WHITE),
            ("current", "  Current: --", FONT_BODY, COLOR_WHITE),
            ("will_add", "  Will add: --", FONT_BODY, COLOR_WHITE),
            ("sep", "  " + "-" * 22, FONT_BODY, COLOR_DIM),
            ("tank_laps", "  Laps on tank: --", FONT_BODY, COLOR_WHITE),
            ("status", "  Waiting...", FONT_STATUS, COLOR_DIM),
        ]
        for key, text, font, color in rows:
            lbl = tk.Label(self._frame, text=text, font=font, fg=color, bg=BG,
                           anchor="w", justify="left")
            lbl.pack(fill="x", padx=4, pady=0)
            lbl.bind("<Button-1>", self._on_drag_start)
            lbl.bind("<B1-Motion>", self._on_drag_motion)
            lbl.bind("<ButtonRelease-1>", self._on_drag_release)
            self._labels[key] = lbl

    def _on_drag_start(self, event) -> None:
        self._drag_x = event.x_root - self._root.winfo_x()
        self._drag_y = event.y_root - self._root.winfo_y()

    def _on_drag_motion(self, event) -> None:
        x = event.x_root - self._drag_x
        y = event.y_root - self._drag_y
        self._root.geometry(f"+{x}+{y}")

    def _on_drag_release(self, event) -> None:
        x = self._root.winfo_x()
        y = self._root.winfo_y()
        self._cfg["overlay_position"]["x"] = x
        self._cfg["overlay_position"]["y"] = y
        if self._on_drag_end:
            self._on_drag_end(x, y)

    def _refresh(self) -> None:
        if not self._running:
            return

        state = self._get_state()
        flash = self._get_flash()

        lbl = self._labels
        s = state.state

        # Flash override
        if flash:
            bg = COLOR_FLASH
            for k in lbl:
                lbl[k].configure(bg=bg)
            self._frame.configure(bg=bg)
            self._root.configure(bg=bg)
            lbl["status"].configure(text=f"  *** {flash} ***", fg=BG)
            self._root.after(100, self._refresh)
            return
        else:
            for k in lbl:
                lbl[k].configure(bg=BG)
            self._frame.configure(bg=BG)
            self._root.configure(bg=BG)

        if s == WAITING:
            lbl["status"].configure(text="  Waiting for iRacing...", fg=COLOR_DIM)
            for k in ["avg_lap", "remaining", "est_laps", "fuel_need", "current", "will_add", "tank_laps"]:
                lbl[k].configure(text=f"  {k.replace('_', ' ').title()}: --", fg=COLOR_DIM)
        elif s == PRACTICE:
            lbl["status"].configure(text="  Practice / Quali", fg=COLOR_DIM)
            _update_data_labels(lbl, state)
        elif s == RACE_INSUFFICIENT_DATA:
            n = state.clean_lap_count
            mn = self._cfg["min_laps_before_auto"]
            lbl["status"].configure(text=f"  Learning... ({n}/{mn} laps)", fg=COLOR_YELLOW)
            _update_data_labels(lbl, state)
        elif s == RACE_READY:
            lbl["status"].configure(text="  Race — Auto-fuel ACTIVE", fg=COLOR_GREEN)
            _update_data_labels(lbl, state)
        elif s == PITTING:
            lbl["status"].configure(text=f"  PITTING — Set: {state.last_pit_fuel_set:.1f}L", fg=COLOR_FLASH)
            _update_data_labels(lbl, state)
        elif s == FINISHED:
            lbl["status"].configure(text="  Session finished", fg=COLOR_DIM)

        # Fuel color for key fields
        laps = state.laps_on_current_tank
        fc = self._fuel_color(laps)
        lbl["tank_laps"].configure(fg=fc)
        lbl["current"].configure(fg=fc)

        # Red flash if critical
        if laps < 1 and s in (RACE_READY, PITTING):
            now_vis = int(time.time() * 2) % 2 == 0
            color = COLOR_RED if now_vis else BG
            lbl["current"].configure(fg=color)
            lbl["tank_laps"].configure(fg=color)

        self._root.after(500, self._refresh)

    def run(self) -> None:
        self._root = tk.Tk()
        self._running = True
        self._build_ui()
        self._refresh()
        self._root.mainloop()
        self._running = False

    def stop(self) -> None:
        self._running = False
        if self._root:
            try:
                self._root.quit()
            except Exception:
                pass


def _update_data_labels(lbl: dict, state: FuelState) -> None:
    avg = state.avg_fuel_per_lap
    lbl["avg_lap"].configure(
        text=f"  Avg/lap:    {avg:.2f} L" if avg else "  Avg/lap:    --",
        fg=COLOR_WHITE,
    )
    lbl["remaining"].configure(
        text=f"  Remaining:  {_fmt_time(state.session_time_remain)}",
        fg=COLOR_WHITE,
    )
    est = state.est_laps_remaining
    lbl["est_laps"].configure(
        text=f"  Est laps:   {est:.1f}" if est else "  Est laps:   --",
        fg=COLOR_WHITE,
    )
    fn = state.fuel_needed
    lbl["fuel_need"].configure(
        text=f"  Fuel need:  {fn:.1f} L" if fn else "  Fuel need:  --",
        fg=COLOR_WHITE,
    )
    lbl["current"].configure(
        text=f"  Current:    {state.current_fuel:.1f} L",
        fg=COLOR_WHITE,
    )
    fa = state.fuel_to_add
    marker = " ✓" if fa > 0 else ""
    lbl["will_add"].configure(
        text=f"  Will add:   {fa:.1f} L{marker}" if avg else "  Will add:   --",
        fg=COLOR_GREEN if fa > 0 else COLOR_DIM,
    )
    laps = state.laps_on_current_tank
    lbl["tank_laps"].configure(
        text=f"  Laps on tank: {laps:.1f}" if avg else "  Laps on tank: --",
        fg=COLOR_WHITE,
    )
