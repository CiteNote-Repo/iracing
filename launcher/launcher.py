#!/usr/bin/env python3
"""
iRacing Tools Launcher — dark-themed tkinter GUI.

Launches iracing_fuel and iracing_live_grip as background subprocesses.
Creates a Windows desktop shortcut on first run.
"""

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# ── Paths ─────────────────────────────────────────────────────────────────────
LAUNCHER_DIR = Path(__file__).parent
ROOT_DIR     = LAUNCHER_DIR.parent
FUEL_DIR     = ROOT_DIR / "iracing_fuel"
GRIP_DIR     = ROOT_DIR / "iracing_live_grip"
GRIP_CFG     = GRIP_DIR / "live_grip_config.json"
LAUNCHER_CFG = LAUNCHER_DIR / "launcher_config.json"

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = "#0d1117"
BG_MID = "#1a1a2e"
BG_SEP = "#2c3e50"
FG     = "#eeeeee"
FG_DIM = "#888888"
ACCENT = "#3498db"
GREEN  = "#2ecc71"
RED    = "#e74c3c"

DEFAULT_CFG: dict = {
    "car":          "Default",
    "volume":       70,
    "alpha":        85,
    "audio_on":     True,
    "fuel_enabled": True,
    "grip_enabled": True,
    "audio_only":   False,
}

# ── Config I/O ────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    if LAUNCHER_CFG.exists():
        try:
            with open(LAUNCHER_CFG) as f:
                return {**DEFAULT_CFG, **json.load(f)}
        except Exception:
            pass
    return dict(DEFAULT_CFG)


def _save_cfg(cfg: dict) -> None:
    with open(LAUNCHER_CFG, "w") as f:
        json.dump(cfg, f, indent=2)


def _car_names() -> list:
    names = ["Default"]
    if GRIP_CFG.exists():
        try:
            with open(GRIP_CFG) as f:
                names += list(json.load(f).get("car_overrides", {}).keys())
        except Exception:
            pass
    return names


# ── Desktop shortcut (Windows only) ──────────────────────────────────────────

def _create_shortcut() -> None:
    if sys.platform != "win32":
        return

    bat = ROOT_DIR / "launch_iracing_tools.bat"
    bat.write_text(
        "@echo off\n"
        f'cd /d "{LAUNCHER_DIR}"\n'
        "pythonw launcher.py\n"
    )

    desktop  = Path.home() / "Desktop"
    lnk_path = desktop / "iRacing Tools.lnk"
    vbs_path = LAUNCHER_DIR / "_shortcut_tmp.vbs"

    vbs_path.write_text(
        f'Set oWS = WScript.CreateObject("WScript.Shell")\n'
        f'sLinkFile = "{lnk_path}"\n'
        f'Set oLink = oWS.CreateShortcut(sLinkFile)\n'
        f'oLink.TargetPath = "{bat}"\n'
        f'oLink.WindowStyle = 7\n'
        f'oLink.Description = "iRacing Tools Launcher"\n'
        f'oLink.WorkingDirectory = "{LAUNCHER_DIR}"\n'
        f'oLink.Save\n'
    )
    try:
        subprocess.run(["cscript", "//nologo", str(vbs_path)],
                       check=True, timeout=10, capture_output=True)
        print(f"Shortcut created: {lnk_path}")
    except Exception as exc:
        print(f"Could not create desktop shortcut: {exc}")
    finally:
        if vbs_path.exists():
            vbs_path.unlink()


# ── Launcher GUI ──────────────────────────────────────────────────────────────

class Launcher:
    def __init__(self) -> None:
        self._cfg   = _load_cfg()
        self._cars  = _car_names()
        self._procs: list[subprocess.Popen] = []

        self._root = tk.Tk()
        self._root.title("iRacing Tools")
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Style combobox to match dark theme
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox",
                        background=BG_MID, foreground=FG,
                        fieldbackground=BG_MID, selectbackground=ACCENT,
                        selectforeground=FG, arrowcolor=FG)

        self._build()
        self._root.after(2000, self._poll_procs)

    # ── UI construction ───────────────────────────────────────────────────────

    def _sep(self) -> None:
        tk.Frame(self._root, bg=BG_SEP, height=1).pack(fill="x", padx=16, pady=6)

    def _build(self) -> None:
        root = self._root
        px   = {"padx": 16, "pady": 3}

        # Title
        tk.Label(root, text="iRacing Tools", font=("Segoe UI", 14, "bold"),
                 bg=BG, fg=FG).pack(pady=(16, 4))

        self._sep()

        # Car selector
        row = tk.Frame(root, bg=BG)
        row.pack(fill="x", **px)
        tk.Label(row, text="Car:", font=("Segoe UI", 10), bg=BG, fg=FG_DIM,
                 width=9, anchor="w").pack(side="left")
        self._car_var = tk.StringVar(value=self._cfg["car"])
        cb = ttk.Combobox(row, textvariable=self._car_var, values=self._cars,
                           state="readonly", width=26, font=("Segoe UI", 10))
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._save())

        self._sep()

        # Tool checkboxes
        self._fuel_var  = tk.BooleanVar(value=self._cfg["fuel_enabled"])
        self._grip_var  = tk.BooleanVar(value=self._cfg["grip_enabled"])
        self._aonly_var = tk.BooleanVar(value=self._cfg["audio_only"])

        for var, label in [
            (self._fuel_var,  "Fuel Calculator"),
            (self._grip_var,  "Live Grip Monitor"),
            (self._aonly_var, "Grip Audio Only (no overlay)"),
        ]:
            f = tk.Frame(root, bg=BG)
            f.pack(fill="x", padx=16, pady=2)
            tk.Checkbutton(f, text=label, variable=var,
                           font=("Segoe UI", 10),
                           bg=BG, fg=FG, selectcolor=BG_MID,
                           activebackground=BG, activeforeground=FG,
                           command=self._save).pack(side="left")

        self._sep()

        # Volume slider
        self._vol_var = tk.IntVar(value=self._cfg["volume"])
        self._vol_lbl = self._slider_row("Volume:", self._vol_var, 0, 100, self._on_vol)

        # Overlay opacity slider
        self._alpha_var = tk.IntVar(value=self._cfg["alpha"])
        self._alpha_lbl = self._slider_row("Overlay:", self._alpha_var, 10, 100, self._on_alpha)

        self._sep()

        # Audio radio buttons
        row = tk.Frame(root, bg=BG)
        row.pack(fill="x", **px)
        tk.Label(row, text="Audio:", font=("Segoe UI", 10), bg=BG, fg=FG_DIM,
                 width=9, anchor="w").pack(side="left")
        self._audio_var = tk.BooleanVar(value=self._cfg["audio_on"])
        for label, val in [("On", True), ("Off", False)]:
            tk.Radiobutton(row, text=f"● {label}" if val else f"○ {label}",
                           variable=self._audio_var, value=val,
                           font=("Segoe UI", 10),
                           bg=BG, fg=FG, selectcolor=BG_MID,
                           activebackground=BG, activeforeground=FG,
                           indicatoron=True,
                           command=self._save).pack(side="left", padx=(0, 8))

        self._sep()

        # Start / Stop button
        self._btn = tk.Button(root, text="  START TOOLS  ",
                               font=("Segoe UI", 11, "bold"),
                               bg=GREEN, fg="white",
                               activebackground="#27ae60", activeforeground="white",
                               bd=0, padx=24, pady=8,
                               cursor="hand2",
                               command=self._toggle)
        self._btn.pack(pady=10)

        # Status line
        self._status = tk.StringVar(value="Status: Waiting...")
        tk.Label(root, textvariable=self._status,
                 font=("Segoe UI", 9), bg=BG, fg=FG_DIM).pack(pady=(0, 16))

        # Fix window width
        root.update_idletasks()
        root.minsize(380, root.winfo_reqheight())

    def _slider_row(self, label: str, var: tk.IntVar,
                    lo: int, hi: int, cmd) -> tk.Label:
        row = tk.Frame(self._root, bg=BG)
        row.pack(fill="x", padx=16, pady=3)
        tk.Label(row, text=label, font=("Segoe UI", 10), bg=BG, fg=FG_DIM,
                 width=9, anchor="w").pack(side="left")
        lbl = tk.Label(row, text=f"{var.get()}%",
                       font=("Segoe UI", 10), bg=BG, fg=FG, width=5)
        lbl.pack(side="right")
        tk.Scale(row, from_=lo, to=hi, orient="horizontal", variable=var,
                 showvalue=False, bg=BG, fg=ACCENT, troughcolor=BG_MID,
                 highlightbackground=BG, bd=0,
                 command=cmd).pack(side="left", fill="x", expand=True)
        return lbl

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_vol(self, v: str) -> None:
        self._vol_lbl.configure(text=f"{int(float(v))}%")
        self._save()

    def _on_alpha(self, v: str) -> None:
        self._alpha_lbl.configure(text=f"{int(float(v))}%")
        self._save()

    def _save(self) -> None:
        self._cfg.update({
            "car":          self._car_var.get(),
            "volume":       self._vol_var.get(),
            "alpha":        self._alpha_var.get(),
            "audio_on":     self._audio_var.get(),
            "fuel_enabled": self._fuel_var.get(),
            "grip_enabled": self._grip_var.get(),
            "audio_only":   self._aonly_var.get(),
        })
        _save_cfg(self._cfg)

    def _toggle(self) -> None:
        if self._procs:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        car      = self._car_var.get()
        vol      = self._vol_var.get() / 100.0
        alpha    = self._alpha_var.get() / 100.0
        audio_on = self._audio_var.get()
        python   = sys.executable
        launched: list[str] = []

        # Fuel Calculator
        if self._fuel_var.get():
            cmd = [python, "main.py", "--alpha", f"{alpha:.2f}"]
            try:
                self._procs.append(subprocess.Popen(cmd, cwd=str(FUEL_DIR)))
                launched.append("Fuel Calculator")
            except Exception as exc:
                self._status.set(f"Error launching fuel: {exc}")
                return

        # Live Grip Monitor (normal or audio-only)
        if self._grip_var.get() or self._aonly_var.get():
            cmd = [python, "main.py",
                   "--alpha",  f"{alpha:.2f}",
                   "--volume", f"{vol:.2f}"]
            if car != "Default":
                cmd += ["--car", car]
            if not audio_on:
                cmd += ["--no-audio"]
            if self._aonly_var.get():
                cmd += ["--audio-only"]
            try:
                self._procs.append(subprocess.Popen(cmd, cwd=str(GRIP_DIR)))
                launched.append("Grip (Audio Only)" if self._aonly_var.get()
                                else "Live Grip Monitor")
            except Exception as exc:
                self._status.set(f"Error launching grip: {exc}")
                return

        if launched:
            self._btn.configure(text="  STOP ALL  ",
                                 bg=RED, activebackground="#c0392b")
            self._status.set("Running: " + ", ".join(launched))
        else:
            self._status.set("Status: No tools selected")

    def _stop(self) -> None:
        for p in self._procs:
            try:
                p.kill()
            except Exception:
                pass
        self._procs.clear()
        self._btn.configure(text="  START TOOLS  ",
                             bg=GREEN, activebackground="#27ae60")
        self._status.set("Status: Stopped")

    def _poll_procs(self) -> None:
        """Check every 2 s whether launched processes are still alive."""
        if self._procs:
            alive = [p for p in self._procs if p.poll() is None]
            if len(alive) < len(self._procs):
                self._procs[:] = alive
                if not self._procs:
                    self._btn.configure(text="  START TOOLS  ",
                                         bg=GREEN, activebackground="#27ae60")
                    self._status.set("Status: All tools exited")
        self._root.after(2000, self._poll_procs)

    def _on_close(self) -> None:
        self._stop()
        self._root.destroy()

    def run(self) -> None:
        self._root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    first_run = not LAUNCHER_CFG.exists()
    app = Launcher()
    if first_run:
        try:
            _create_shortcut()
        except Exception as exc:
            print(f"Could not create shortcut: {exc}")
    app.run()


if __name__ == "__main__":
    main()
