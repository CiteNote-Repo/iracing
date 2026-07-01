#!/usr/bin/env python3
"""
iRacing Tools Launcher — dark-themed tkinter GUI.

Tab 1: Live Tools — launches iracing_fuel and iracing_live_grip as background subprocesses.
Tab 2: Analyzer  — post-session bilateral axle balance analysis.
"""

import json
import subprocess
import sys
import threading
import webbrowser
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

# Try drag-and-drop support (optional dependency)
try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    _HAS_DND = True
except ImportError:
    _HAS_DND = False

# ── Paths ─────────────────────────────────────────────────────────────────────
LAUNCHER_DIR  = Path(__file__).parent
ROOT_DIR      = LAUNCHER_DIR.parent
FUEL_DIR      = ROOT_DIR / "iracing_fuel"
GRIP_DIR      = ROOT_DIR / "iracing_live_grip"
ANALYZER_DIR  = ROOT_DIR / "bilateral_analyzer"
GRIP_CFG      = GRIP_DIR / "live_grip_config.json"
BILATERAL_CFG = ANALYZER_DIR / "bilateral_config.json"
LAUNCHER_CFG  = LAUNCHER_DIR / "launcher_config.json"

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
    "recent_ibt":   [],
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


def _analyzer_car_names() -> list:
    names = ["Auto-detect"]
    if BILATERAL_CFG.exists():
        try:
            with open(BILATERAL_CFG) as f:
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
        self._analyzing = False
        self._dot_count = 0

        if _HAS_DND:
            self._root = TkinterDnD.Tk()
        else:
            self._root = tk.Tk()

        self._root.title("iRacing Tools")
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox",
                        background=BG_MID, foreground=FG,
                        fieldbackground=BG_MID, selectbackground=ACCENT,
                        selectforeground=FG, arrowcolor=FG)
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_MID, foreground=FG_DIM,
                        padding=[14, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", FG)])

        self._build()
        self._root.after(2000, self._poll_procs)

    # ── UI construction ───────────────────────────────────────────────────────

    def _sep(self, parent: tk.Widget) -> None:
        tk.Frame(parent, bg=BG_SEP, height=1).pack(fill="x", padx=16, pady=6)

    def _build(self) -> None:
        root = self._root

        tk.Label(root, text="iRacing Tools", font=("Segoe UI", 14, "bold"),
                 bg=BG, fg=FG).pack(pady=(16, 8))

        nb = ttk.Notebook(root)
        nb.pack(fill="both", expand=True)

        live_frame     = tk.Frame(nb, bg=BG)
        analyzer_frame = tk.Frame(nb, bg=BG)
        nb.add(live_frame,     text="  Live Tools  ")
        nb.add(analyzer_frame, text="  Analyzer  ")

        self._build_live_tab(live_frame)
        self._build_analyzer_tab(analyzer_frame)

        root.update_idletasks()
        root.minsize(380, root.winfo_reqheight())

    def _build_live_tab(self, frame: tk.Frame) -> None:
        px = {"padx": 16, "pady": 3}

        self._sep(frame)

        # Car selector
        row = tk.Frame(frame, bg=BG)
        row.pack(fill="x", **px)
        tk.Label(row, text="Car:", font=("Segoe UI", 10), bg=BG, fg=FG_DIM,
                 width=9, anchor="w").pack(side="left")
        self._car_var = tk.StringVar(value=self._cfg["car"])
        cb = ttk.Combobox(row, textvariable=self._car_var, values=self._cars,
                           state="readonly", width=26, font=("Segoe UI", 10))
        cb.pack(side="left", fill="x", expand=True)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._save())

        self._sep(frame)

        # Tool checkboxes
        self._fuel_var  = tk.BooleanVar(value=self._cfg["fuel_enabled"])
        self._grip_var  = tk.BooleanVar(value=self._cfg["grip_enabled"])
        self._aonly_var = tk.BooleanVar(value=self._cfg["audio_only"])

        for var, label in [
            (self._fuel_var,  "Fuel Calculator"),
            (self._grip_var,  "Live Grip Monitor"),
            (self._aonly_var, "Grip Audio Only (no overlay)"),
        ]:
            f = tk.Frame(frame, bg=BG)
            f.pack(fill="x", padx=16, pady=2)
            tk.Checkbutton(f, text=label, variable=var,
                           font=("Segoe UI", 10),
                           bg=BG, fg=FG, selectcolor=BG_MID,
                           activebackground=BG, activeforeground=FG,
                           command=self._save).pack(side="left")

        self._sep(frame)

        # Volume slider
        self._vol_var = tk.IntVar(value=self._cfg["volume"])
        self._vol_lbl = self._slider_row(frame, "Volume:", self._vol_var, 0, 100, self._on_vol)

        # Overlay opacity slider
        self._alpha_var = tk.IntVar(value=self._cfg["alpha"])
        self._alpha_lbl = self._slider_row(frame, "Overlay:", self._alpha_var, 10, 100, self._on_alpha)

        self._sep(frame)

        # Audio radio buttons
        row = tk.Frame(frame, bg=BG)
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

        self._sep(frame)

        # Start / Stop button
        self._btn = tk.Button(frame, text="  START TOOLS  ",
                               font=("Segoe UI", 11, "bold"),
                               bg=GREEN, fg="white",
                               activebackground="#27ae60", activeforeground="white",
                               bd=0, padx=24, pady=8,
                               cursor="hand2",
                               command=self._toggle)
        self._btn.pack(pady=10)

        # Status line
        self._status = tk.StringVar(value="Status: Waiting...")
        tk.Label(frame, textvariable=self._status,
                 font=("Segoe UI", 9), bg=BG, fg=FG_DIM).pack(pady=(0, 16))

    def _build_analyzer_tab(self, frame: tk.Frame) -> None:
        self._ibt_path  = tk.StringVar(value="")
        self._az_status = tk.StringVar(value="Status: Ready")
        self._az_car_var = tk.StringVar(value="Auto-detect")

        # Drop zone — thin border frame wrapping a label
        drop_outer = tk.Frame(frame, bg=BG_SEP, padx=1, pady=1)
        drop_outer.pack(fill="x", padx=16, pady=(16, 8))
        self._drop_zone = tk.Label(
            drop_outer,
            text="Drop .ibt file here\nor click to browse",
            font=("Segoe UI", 10),
            bg=BG_MID, fg=FG_DIM,
            height=4, cursor="hand2",
        )
        self._drop_zone.pack(fill="both")
        self._drop_zone.bind("<Button-1>", lambda _e: self._browse_ibt())

        if _HAS_DND:
            self._drop_zone.drop_target_register(DND_FILES)
            self._drop_zone.dnd_bind("<<Drop>>",      self._on_drop)
            self._drop_zone.dnd_bind("<<DragEnter>>", self._on_drag_enter)
            self._drop_zone.dnd_bind("<<DragLeave>>", self._on_drag_leave)
        else:
            tk.Label(frame, text="Install tkinterdnd2 for drag-and-drop support",
                     font=("Segoe UI", 8), bg=BG, fg=FG_DIM).pack()

        # Car selector
        row = tk.Frame(frame, bg=BG)
        row.pack(fill="x", padx=16, pady=(4, 4))
        tk.Label(row, text="Car:", font=("Segoe UI", 10), bg=BG, fg=FG_DIM,
                 width=9, anchor="w").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self._az_car_var,
                          values=_analyzer_car_names(),
                          state="readonly", width=26, font=("Segoe UI", 10))
        cb.pack(side="left", fill="x", expand=True)

        self._sep(frame)

        # Recent files
        tk.Label(frame, text="Recent:", font=("Segoe UI", 9),
                 bg=BG, fg=FG_DIM, anchor="w").pack(fill="x", padx=16)
        self._recent_frame = tk.Frame(frame, bg=BG)
        self._recent_frame.pack(fill="x", padx=16, pady=(2, 0))
        self._refresh_recent()

        self._sep(frame)

        # Analyze button
        self._az_btn = tk.Button(frame, text="  ANALYZE  ",
                                  font=("Segoe UI", 11, "bold"),
                                  bg=ACCENT, fg="white",
                                  activebackground="#2980b9", activeforeground="white",
                                  bd=0, padx=24, pady=8,
                                  cursor="hand2", state="disabled",
                                  command=self._run_analysis)
        self._az_btn.pack(pady=10)

        # Status
        tk.Label(frame, textvariable=self._az_status,
                 font=("Segoe UI", 9), bg=BG, fg=FG_DIM).pack(pady=(0, 16))

    def _refresh_recent(self) -> None:
        for w in self._recent_frame.winfo_children():
            w.destroy()
        recent = self._cfg.get("recent_ibt", [])
        if not recent:
            tk.Label(self._recent_frame, text="  (none)", font=("Segoe UI", 9),
                     bg=BG, fg=FG_DIM).pack(anchor="w")
            return
        for path in recent[:5]:
            name  = Path(path).name
            short = name[:46] + ("…" if len(name) > 46 else "")
            lbl   = tk.Label(self._recent_frame, text=f"  › {short}",
                             font=("Segoe UI", 9), bg=BG, fg=ACCENT,
                             cursor="hand2", anchor="w")
            lbl.pack(fill="x")
            lbl.bind("<Button-1>", lambda _e, p=path: self._load_ibt(p))

    def _slider_row(self, parent: tk.Widget, label: str, var: tk.IntVar,
                    lo: int, hi: int, cmd) -> tk.Label:
        row = tk.Frame(parent, bg=BG)
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

    # ── Drag-and-drop handlers ────────────────────────────────────────────────

    def _on_drop(self, event) -> None:
        raw = event.data.strip()
        # tkinterdnd2 wraps paths with spaces in braces on some platforms
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        self._load_ibt(raw)
        self._drop_zone.configure(bg=BG_MID, fg=FG_DIM)

    def _on_drag_enter(self, _event) -> None:
        self._drop_zone.configure(bg=BG_SEP, fg=FG)

    def _on_drag_leave(self, _event) -> None:
        self._drop_zone.configure(bg=BG_MID, fg=FG_DIM)

    def _browse_ibt(self) -> None:
        path = filedialog.askopenfilename(
            title="Select .ibt file",
            filetypes=[("iRacing Telemetry", "*.ibt"), ("All files", "*.*")],
            initialdir=str(Path.home() / "Documents" / "iRacing" / "telemetry"),
        )
        if path:
            self._load_ibt(path)

    def _load_ibt(self, path: str) -> None:
        self._ibt_path.set(path)
        self._drop_zone.configure(text=Path(path).name, fg=FG)
        self._az_btn.configure(state="normal")
        self._az_status.set("Status: Ready")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _run_analysis(self) -> None:
        path = self._ibt_path.get()
        if not path or not Path(path).exists():
            self._az_status.set("Error: file not found")
            return
        if self._analyzing:
            return
        self._analyzing = True
        self._az_btn.configure(state="disabled")
        self._dot_count = 0
        self._animate_dots()
        car = self._az_car_var.get()
        threading.Thread(
            target=self._analysis_worker, args=(path, car), daemon=True
        ).start()

    def _animate_dots(self) -> None:
        if not self._analyzing:
            return
        dots = "." * (self._dot_count % 4)
        self._az_status.set(f"Analyzing{dots}")
        self._dot_count += 1
        self._root.after(500, self._animate_dots)

    def _analysis_worker(self, ibt_path: str, car_name: str) -> None:
        cmd = [sys.executable, "main.py", "--file", ibt_path, "--no-browser"]
        if car_name != "Auto-detect":
            # Translate car name to explicit param overrides (no --car flag in analyzer)
            try:
                with open(BILATERAL_CFG) as f:
                    overrides = json.load(f).get("car_overrides", {}).get(car_name, {})
                if "wheelbase_m" in overrides:
                    cmd += ["--wheelbase", str(overrides["wheelbase_m"])]
                if "steering_ratio" in overrides:
                    cmd += ["--steering-ratio", str(overrides["steering_ratio"])]
            except Exception:
                pass
        result = subprocess.run(cmd, cwd=str(ANALYZER_DIR), capture_output=True, text=True)
        self._root.after(0, lambda: self._analysis_done(ibt_path, result))

    def _analysis_done(self, ibt_path: str, result: subprocess.CompletedProcess) -> None:
        self._analyzing = False
        self._az_btn.configure(state="normal")

        stem      = Path(ibt_path).stem
        html_path = ANALYZER_DIR / f"bilateral_{stem}.html"
        if html_path.exists():
            webbrowser.open(str(html_path))
            self._az_status.set("Done — report opened in browser")
            self._add_recent(ibt_path)
        else:
            err = (result.stderr or result.stdout or "unknown error")[:200]
            self._az_status.set(f"Error: {err}")

    def _add_recent(self, path: str) -> None:
        recent = self._cfg.get("recent_ibt", [])
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._cfg["recent_ibt"] = recent[:5]
        _save_cfg(self._cfg)
        self._refresh_recent()

    # ── Event handlers (Live Tools tab) ──────────────────────────────────────

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

        if self._fuel_var.get():
            cmd = [python, "main.py", "--alpha", f"{alpha:.2f}"]
            try:
                self._procs.append(subprocess.Popen(cmd, cwd=str(FUEL_DIR)))
                launched.append("Fuel Calculator")
            except Exception as exc:
                self._status.set(f"Error launching fuel: {exc}")
                return

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
