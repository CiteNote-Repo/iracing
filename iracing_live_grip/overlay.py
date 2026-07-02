import threading
import tkinter as tk
from typing import Callable, Optional

from grip_calculator import GripData, QUIET, PEAK_GRIP_HISS, REAR_SLIDE, OVERSTEER_HOWL

BG = "black"
FONT_TITLE = ("Consolas", 10, "bold")
FONT_BAR = ("Consolas", 9)
FONT_META = ("Consolas", 8)
FONT_STATE = ("Consolas", 8, "italic")
COLOR_WHITE = "#FFFFFF"
COLOR_DIM = "#666666"
COLOR_ORANGE = "#E67E22"
COLOR_RED = "#E74C3C"
COLOR_BLUE = "#3498DB"

_STATE_COLORS = {
    QUIET:          "#555555",
    PEAK_GRIP_HISS: "#2ECC71",
    REAR_SLIDE:     "#E67E22",
    OVERSTEER_HOWL: "#E74C3C",
}

_BAR_CHARS = 20  # width of total-util progress bar


def _make_bar(pct: float) -> str:
    # Scale 0-130% linearly so there is headroom above 100%
    fill = max(0, min(int(min(pct, 130.0) / 130.0 * _BAR_CHARS), _BAR_CHARS))
    return "█" * fill + "░" * (_BAR_CHARS - fill)


def _state_color(state: str) -> str:
    return _STATE_COLORS.get(state, COLOR_DIM)


class GripOverlay:
    def __init__(
        self,
        cfg: dict,
        get_grip_fn: Callable[[], GripData],
        on_drag_end: Optional[Callable] = None,
    ):
        self._cfg = cfg
        self._get_grip = get_grip_fn
        self._on_drag_end = on_drag_end
        self._root: Optional[tk.Tk] = None
        self._running = False
        self._drag_x = 0
        self._drag_y = 0

    # ── drag support ──────────────────────────────────────────────────────────

    def _bind_drag(self, widget) -> None:
        widget.bind("<Button-1>", self._on_drag_start)
        widget.bind("<B1-Motion>", self._on_drag_motion)
        widget.bind("<ButtonRelease-1>", self._on_drag_release)

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

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = self._root
        pos = self._cfg["overlay_position"]
        alpha = self._cfg.get("overlay_alpha", 0.85)

        root.geometry(f"+{pos['x']}+{pos['y']}")
        root.attributes("-topmost", True)
        root.attributes("-alpha", alpha)
        root.overrideredirect(True)
        root.configure(bg=BG)
        try:
            root.wm_attributes("-transparentcolor", BG)
        except Exception:
            pass

        self._bind_drag(root)

        frame = tk.Frame(root, bg=BG, bd=1, relief="solid",
                         highlightbackground="#444444", highlightthickness=1)
        frame.pack(padx=2, pady=2)
        self._bind_drag(frame)

        def row(text, font, fg):
            lbl = tk.Label(frame, text=text, font=font, fg=fg, bg=BG,
                           anchor="w", justify="left")
            lbl.pack(fill="x", padx=4, pady=0)
            self._bind_drag(lbl)
            return lbl

        row("  GRIP MONITOR  ", FONT_TITLE, COLOR_WHITE)

        blank_bar = "░" * _BAR_CHARS
        self._lbl_util = row(f"  {blank_bar}   --%", FONT_BAR, COLOR_DIM)

        row("  " + "─" * 28, FONT_META, "#333333")

        self._lbl_scrub = row("  Front scrub:  --", FONT_META, COLOR_DIM)
        self._lbl_yaw_dev = row("  Yaw dev:      --", FONT_META, COLOR_DIM)

        row("  " + "─" * 28, FONT_META, "#333333")

        self._lbl_state = row("  Waiting for iRacing...  ", FONT_STATE, COLOR_DIM)

    # ── refresh loop (20 Hz) ──────────────────────────────────────────────────

    def _refresh(self) -> None:
        if not self._running:
            return

        grip = self._get_grip()

        if not grip.connected:
            blank = "░" * _BAR_CHARS
            self._lbl_util.configure(text=f"  {blank}   --%", fg=COLOR_DIM)
            self._lbl_scrub.configure(text="  Front scrub:  --", fg=COLOR_DIM)
            self._lbl_yaw_dev.configure(text="  Yaw dev:      --", fg=COLOR_DIM)
            self._lbl_state.configure(text="  Waiting for iRacing...  ", fg=COLOR_DIM)
        else:
            active = grip.is_on_track and grip.speed_mps >= 5.0
            util_color = _state_color(grip.overall_state) if active else COLOR_DIM

            util_bar = _make_bar(grip.total_util)
            self._lbl_util.configure(
                text=f"  {util_bar}  {grip.total_util:3.0f}%",
                fg=util_color,
            )

            if grip.scrub_proximity_pct > 0:
                scrub_color = (
                    COLOR_RED if grip.scrub_proximity_pct < 60.0 else
                    COLOR_ORANGE if grip.scrub_proximity_pct < 80.0 else
                    COLOR_WHITE
                ) if active else COLOR_DIM
                self._lbl_scrub.configure(
                    text=f"  Front scrub: {grip.scrub_proximity_pct:3.0f}%",
                    fg=scrub_color,
                )
            else:
                self._lbl_scrub.configure(
                    text="  Front scrub:  --",
                    fg=COLOR_DIM,
                )

            yaw = grip.yaw_deviation_pct
            if abs(yaw) >= 1.0:
                sign = "+" if yaw >= 0 else ""
                if yaw > 0:
                    yaw_color = COLOR_RED if yaw > 30.0 else COLOR_ORANGE
                else:
                    yaw_color = COLOR_BLUE
                self._lbl_yaw_dev.configure(
                    text=f"  Yaw dev: {sign}{yaw:3.0f}%",
                    fg=yaw_color if active else COLOR_DIM,
                )
            else:
                self._lbl_yaw_dev.configure(
                    text="  Yaw dev:      --",
                    fg=COLOR_DIM,
                )

            state_color = _state_color(grip.overall_state) if active else COLOR_DIM
            state_label = grip.overall_state.replace("_", " ")
            self._lbl_state.configure(
                text=f"  State: {state_label:<18}",
                fg=state_color,
            )

        self._root.after(50, self._refresh)

    # ── lifecycle ─────────────────────────────────────────────────────────────

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
