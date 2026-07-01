"""
HTML report generator.  Produces a single self-contained file with tabbed pages.
Plotly JS is loaded from CDN (internet required to view interactive charts).
"""

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from corner_detector import Corner
from tyre_energy_tracker import STATE_COLORS, STATE_NAMES, QUIET


# ── Colour palette ────────────────────────────────────────────────────────────
C_GREEN   = "#2ecc71"
C_YELLOW  = "#f39c12"
C_RED     = "#e74c3c"
C_BLUE    = "#3498db"
C_DARK    = "#2c3e50"
C_GREY    = "#95a5a6"
C_OVERLAY = "rgba(52,152,219,0.15)"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _acoustic_vrects(
    fig: "go.Figure",
    t_rel: np.ndarray,
    state_arr: np.ndarray,
    n_rows: int = 6,
) -> None:
    """Overlay acoustic-state colored background bands on all chart rows."""
    n = len(state_arr)
    if n == 0:
        return
    changes = np.where(np.diff(state_arr) != 0)[0] + 1
    starts  = np.concatenate([[0], changes])
    ends    = np.concatenate([changes, [n]])
    for s_idx, e_idx in zip(starts, ends):
        if e_idx - s_idx < 3:          # skip sub-50 ms flickers
            continue
        state = int(state_arr[s_idx])
        if state == QUIET:
            continue                    # transparent — nothing to draw
        color = STATE_COLORS.get(state, "rgba(0,0,0,0)")
        x0 = float(t_rel[s_idx])
        x1 = float(t_rel[min(int(e_idx), n) - 1])
        if x1 <= x0:
            continue
        for row in range(1, n_rows + 1):
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_width=0,
                          row=row, col=1)


def _div(fig, include_js: bool = False) -> str:
    return fig.to_html(include_plotlyjs="cdn" if include_js else False, full_html=False)


def _fmt_time(t: float) -> str:
    if math.isnan(t):
        return "—"
    m = int(t // 60)
    s = t - m * 60
    return f"{m}:{s:06.3f}"


def _color_bilateral(score: float) -> str:
    if score >= 70:
        return C_GREEN
    if score >= 40:
        return C_YELLOW
    return C_RED


# ── Page builders ─────────────────────────────────────────────────────────────

def _page_overview(lap_summaries: list[dict]) -> str:
    if not lap_summaries:
        return "<p>No lap data found.</p>"

    best_lap = max(lap_summaries, key=lambda r: r["mean_bilateral"])["lap"]
    worst_lap = min(lap_summaries, key=lambda r: r["mean_bilateral"])["lap"]

    rows = ""
    for r in lap_summaries:
        lt = _fmt_time(r["lap_time"])
        bg = ""
        if r["lap"] == best_lap:
            bg = f'style="background:{C_GREEN}22"'
        elif r["lap"] == worst_lap:
            bg = f'style="background:{C_RED}22"'
        util = r.get("mean_lateral_utilization", float("nan"))
        util_str = f"{util:.1f}" if not math.isnan(util) else "—"
        rows += (
            f'<tr {bg}>'
            f'<td>{r["lap"]}</td>'
            f'<td>{lt}</td>'
            f'<td>{r["mean_bilateral"]:.1f}</td>'
            f'<td>{r["peak_bilateral"]:.1f}</td>'
            f'<td>{r["good_corners"]}/{r["total_corners"]}</td>'
            f'<td>{r["rear_led_corners"]}</td>'
            f'<td>{r["front_led_corners"]}</td>'
            f'<td>{r["overlap_quality"]:.1f}</td>'
            f'<td>{util_str}</td>'
            f'</tr>'
        )

    html = f"""
<h2>Session Overview</h2>
<p>Best bilateral lap: <strong>Lap {best_lap}</strong> &nbsp;|&nbsp;
   Worst bilateral lap: <strong>Lap {worst_lap}</strong></p>
<table class="data-table">
<thead><tr>
  <th>Lap</th><th>Time</th>
  <th>Avg Bilateral</th><th>Peak Bilateral</th>
  <th>Good Corners</th><th>Rear-Led</th><th>Front-Led</th>
  <th>Overlap Quality (0-10)</th>
  <th>Mean Lateral Util %</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>"""

    # Bar chart of mean bilateral per lap
    fig = go.Figure()
    laps = [r["lap"] for r in lap_summaries]
    scores = [r["mean_bilateral"] for r in lap_summaries]
    colors = [_color_bilateral(s) for s in scores]
    fig.add_trace(go.Bar(x=laps, y=scores, marker_color=colors, name="Avg Bilateral"))
    fig.add_hline(y=70, line_dash="dash", line_color=C_GREEN, annotation_text="Good threshold (70)")
    fig.update_layout(
        title="Mean Bilateral Score per Lap",
        xaxis_title="Lap",
        yaxis_title="Bilateral Score (0-100)",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font_color="#eee",
        yaxis_range=[0, 100],
    )
    html += _div(fig, include_js=True)

    # ── Acoustic state breakdown ──────────────────────────────────────────────
    # Only render if acoustic data was computed (check for any state key)
    if lap_summaries and "peak_grip_hiss_s" in lap_summaries[0]:
        wheel_ok = not math.isnan(lap_summaries[0].get("peak_rear_slip", float("nan"))) and \
                   any(lap_summaries[0].get("rear_slide_s", 0) > 0
                       for _ in [None])  # True only when slip data present
        slip_note = "" if any(
            r.get("rear_slide_s", 0) + r.get("front_lock_s", 0) > 0.01
            for r in lap_summaries
        ) else " &nbsp;<em style='color:#aaa;font-size:11px'>(wheel speeds not recorded — slip states unavailable)</em>"

        a_rows = ""
        for r in lap_summaries:
            def _s(key: str) -> str:
                v = r.get(key, float("nan"))
                return f"{v:.1f}" if not math.isnan(v) else "—"
            pr = r.get("peak_rear_slip", float("nan"))
            pr_str = f"{pr:.3f}" if not math.isnan(pr) else "—"
            a_rows += (
                f'<tr><td>{r["lap"]}</td>'
                f'<td>{_s("quiet_s")}</td>'
                f'<td>{_s("peak_grip_hiss_s")}</td>'
                f'<td>{_s("rear_slide_s")}</td>'
                f'<td>{_s("front_lock_s")}</td>'
                f'<td>{_s("oversteer_howl_s")}</td>'
                f'<td>{pr_str}</td></tr>'
            )

        html += f"""
<h3 style="margin-top:24px">Acoustic State Breakdown (seconds per lap){slip_note}</h3>
<table class="data-table">
<thead><tr>
  <th>Lap</th>
  <th style="color:{C_GREY}">Quiet (s)</th>
  <th style="color:{C_GREEN}">Peak Grip Hiss (s)</th>
  <th style="color:#e67e22">Rear Slide (s)</th>
  <th style="color:{C_BLUE}">Front Lock (s)</th>
  <th style="color:{C_RED}">Oversteer Howl (s)</th>
  <th>Peak Rear Slip</th>
</tr></thead>
<tbody>{a_rows}</tbody>
</table>"""

    return html


def _page_trackmap(
    df: pd.DataFrame,
    track_x: np.ndarray,
    track_y: np.ndarray,
    best_lap_num: int,
) -> str:
    lap_mask = df["Lap"] == best_lap_num
    x = track_x[lap_mask.to_numpy()]
    y = track_y[lap_mask.to_numpy()]
    bilateral = df.loc[lap_mask, "bilateral_score"].to_numpy()
    glat_abs = df.loc[lap_mask, "gLat"].abs().to_numpy()
    in_corner = glat_abs > 0.3

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="markers",
        marker=dict(
            color=bilateral,
            colorscale=[
                [0.0, C_BLUE],
                [0.3, C_RED],
                [0.6, C_YELLOW],
                [1.0, C_GREEN],
            ],
            cmin=0, cmax=100,
            size=3,
            colorbar=dict(title="Bilateral Score", tickvals=[0, 25, 50, 70, 100]),
        ),
        text=[f"Bilateral: {b:.0f}%" for b in bilateral],
        hovertemplate="%{text}<extra></extra>",
        name="Bilateral Score",
    ))
    fig.update_layout(
        title=f"Track Map — Bilateral Balance (Lap {best_lap_num})",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font_color="#eee",
        height=600,
    )
    return f"<h2>Track Map — Bilateral Balance</h2>{_div(fig)}"


def _corner_chart(df: pd.DataFrame, c: Corner) -> str:
    """Build one 6-row Plotly chart for a single corner. Returns HTML div."""
    s, e = c.entry_idx, c.exit_idx
    seg = df.iloc[s:e + 1].copy()
    t_rel = (seg["time"] - seg["time"].iloc[0]).to_numpy()

    fig = make_subplots(
        rows=6, cols=1, shared_xaxes=True,
        row_heights=[0.17, 0.12, 0.17, 0.17, 0.20, 0.17],
        subplot_titles=[
            "Channel 3 — Brake / Throttle",
            "Channel 1+2 — Steering",
            "Front Slip %",
            "Rear Slip %",
            "Bilateral Score & Imbalance",
            "Absolute Utilization (gLat-based)",
        ],
        vertical_spacing=0.05,
    )

    fig.add_trace(go.Scatter(x=t_rel, y=seg["Brake"],    name="Brake",    line=dict(color=C_RED)),   row=1, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["Throttle"], name="Throttle", line=dict(color=C_GREEN)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["SteeringWheelAngle"], name="Steer", line=dict(color=C_BLUE)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["alpha_front_pct"], name="Front %",  line=dict(color=C_YELLOW)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["alpha_rear_pct"],  name="Rear %",   line=dict(color=C_RED)),    row=4, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["bilateral_score"],         name="Bilateral",   fill="tozeroy", line=dict(color=C_GREEN)),  row=5, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["axle_imbalance"],          name="Imbalance",   line=dict(color=C_GREY, dash="dot")),       row=5, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["understeer_gradient_pct"], name="US Gradient", line=dict(color="#e67e22")),                row=5, col=1)
    fig.add_hline(y=0, line_color="white", line_dash="dot", row=5, col=1)

    fig.add_trace(go.Scatter(x=t_rel, y=seg["lateral_utilization_pct"], name="Total Utilization %",        fill="tozeroy", line=dict(color=C_BLUE)),    row=6, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=seg["axle_balance_pct"],        name="Axle Balance (+=understeer)", line=dict(color="#9b59b6")),                row=6, col=1)
    fig.add_hline(y=0, line_color="white", line_dash="dot", row=6, col=1)

    # Slip ratios scaled ×100 so they share row-6's ±100 axis
    wheel_ok = bool(seg.get("wheel_speeds_available", pd.Series([False])).iloc[0]) if "wheel_speeds_available" in seg.columns else False
    rear_slip_pct  = seg["rear_slip"].to_numpy(float)  * 100 if "rear_slip"  in seg.columns else np.zeros(len(seg))
    front_slip_pct = seg["front_slip"].to_numpy(float) * 100 if "front_slip" in seg.columns else np.zeros(len(seg))
    slip_suffix = "" if wheel_ok else " (no data)"
    fig.add_trace(go.Scatter(x=t_rel, y=rear_slip_pct,  name=f"Rear Slip ×100{slip_suffix}",  line=dict(color=C_YELLOW, dash="dot")), row=6, col=1)
    fig.add_trace(go.Scatter(x=t_rel, y=front_slip_pct, name=f"Front Slip ×100{slip_suffix}", line=dict(color=C_RED,    dash="dot")), row=6, col=1)

    # Acoustic state colored background bands (applied before overlap shade so
    # the overlap shade renders on top with its own distinct blue tint)
    if "acoustic_state" in seg.columns:
        _acoustic_vrects(fig, t_rel, seg["acoustic_state"].to_numpy(int), n_rows=6)

    ov = seg["overlap_active"].to_numpy()
    if ov.any():
        ov_start = t_rel[ov.astype(bool)][0]
        ov_end   = t_rel[ov.astype(bool)][-1]
        for row in range(1, 7):
            fig.add_vrect(x0=ov_start, x1=ov_end, fillcolor=C_OVERLAY, line_width=0, row=row, col=1)

    peak_idx_local = int(seg["bilateral_score"].to_numpy().argmax())
    fig.add_vline(x=t_rel[peak_idx_local], line_color=C_GREEN, line_dash="dot",
                  annotation_text=f"Peak {c.peak_bilateral:.0f}%", annotation_position="top right")

    if not math.isnan(c.throttle_app_time):
        fig.add_vline(x=c.throttle_app_time, line_color=C_YELLOW, line_dash="dash",
                      annotation_text="Throttle", annotation_position="top left")

    fig.update_yaxes(range=[0, 1.1],    row=1, col=1)
    fig.update_yaxes(range=[0, 100],    row=3, col=1)
    fig.update_yaxes(range=[0, 100],    row=4, col=1)
    fig.update_yaxes(range=[-100, 100], row=5, col=1)
    fig.update_yaxes(range=[-100, 110], row=6, col=1)
    fig.update_layout(
        title=f"Lap {c.lap} | Track Corner {c.track_corner_num + 1} | "
              f"Peak bilateral: {c.peak_bilateral:.0f}% | {c.duration:.1f}s",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font_color="#eee",
        height=780,
        showlegend=False,
    )
    return _div(fig)


def _page_corner_deep_dive(
    df: pd.DataFrame,
    corners: list[Corner],
    max_per_lap: int = 5,
) -> str:
    """Show up to max_per_lap evenly-distributed corners per lap."""
    html = "<h2>Corner Deep Dive</h2>"

    laps_in_corners = sorted(set(c.lap for c in corners))

    for lap_num in laps_in_corners:
        lap_corners = [c for c in corners
                       if c.lap == lap_num and c.exit_idx - c.entry_idx >= 10]
        if not lap_corners:
            continue

        html += f'<h3 style="color:#3498db;margin-top:28px">Lap {lap_num}</h3>'

        # Pick evenly-spaced corners so we sample the whole lap
        if len(lap_corners) <= max_per_lap:
            selected = lap_corners
        else:
            step = len(lap_corners) / max_per_lap
            selected = [lap_corners[int(i * step)] for i in range(max_per_lap)]

        for c in selected:
            html += _corner_chart(df, c)

    return html


def _page_best_vs_worst(
    df: pd.DataFrame,
    corners: list[Corner],
    lap_summaries: list[dict],
    tick_rate: int = 60,
) -> str:
    html = "<h2>Best vs Worst Lap — Corner Comparison</h2>"

    if len(lap_summaries) < 2:
        return html + "<p>Need at least 2 laps for comparison.</p>"

    best_lap = max(lap_summaries, key=lambda r: r["mean_bilateral"])["lap"]
    worst_lap = min(lap_summaries, key=lambda r: r["mean_bilateral"])["lap"]

    html += f"<p><strong>Best lap:</strong> Lap {best_lap} &nbsp;|&nbsp; <strong>Worst lap:</strong> Lap {worst_lap}</p>"

    # Group corners by track_corner_num
    from collections import defaultdict
    by_track: dict[int, list[Corner]] = defaultdict(list)
    for c in corners:
        if c.lap in (best_lap, worst_lap):
            by_track[c.track_corner_num].append(c)

    shown = 0
    for track_num, tc_corners in sorted(by_track.items()):
        best_c = next((c for c in tc_corners if c.lap == best_lap), None)
        worst_c = next((c for c in tc_corners if c.lap == worst_lap), None)
        if not best_c or not worst_c:
            continue
        if shown >= 8:
            break

        b_seg = df.iloc[best_c.entry_idx:best_c.exit_idx + 1]
        w_seg = df.iloc[worst_c.entry_idx:worst_c.exit_idx + 1]
        b_t = (b_seg["time"] - b_seg["time"].iloc[0]).to_numpy()
        w_t = (w_seg["time"] - w_seg["time"].iloc[0]).to_numpy()

        fig = make_subplots(rows=4, cols=1, shared_xaxes=False,
                            subplot_titles=["Brake / Throttle", "Steering", "Front/Rear Slip %", "Bilateral Score"],
                            vertical_spacing=0.1)

        # Best lap (solid)
        fig.add_trace(go.Scatter(x=b_t, y=b_seg["Brake"].to_numpy(), name="Best Brake", line=dict(color=C_GREEN)), row=1, col=1)
        fig.add_trace(go.Scatter(x=b_t, y=b_seg["Throttle"].to_numpy(), name="Best Throttle", line=dict(color=C_GREEN, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=b_t, y=np.abs(b_seg["SteeringWheelAngle"].to_numpy()), name="Best Steer", line=dict(color=C_GREEN)), row=2, col=1)
        fig.add_trace(go.Scatter(x=b_t, y=b_seg["alpha_front_pct"].to_numpy(), name="Best Front%", line=dict(color=C_BLUE)), row=3, col=1)
        fig.add_trace(go.Scatter(x=b_t, y=b_seg["alpha_rear_pct"].to_numpy(), name="Best Rear%", line=dict(color=C_YELLOW)), row=3, col=1)
        fig.add_trace(go.Scatter(x=b_t, y=b_seg["bilateral_score"].to_numpy(), name="Best Bilateral", fill="tozeroy", line=dict(color=C_GREEN)), row=4, col=1)

        # Worst lap (dashed)
        fig.add_trace(go.Scatter(x=w_t, y=w_seg["Brake"].to_numpy(), name="Worst Brake", line=dict(color=C_RED, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=w_t, y=w_seg["Throttle"].to_numpy(), name="Worst Throttle", line=dict(color=C_RED, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=w_t, y=np.abs(w_seg["SteeringWheelAngle"].to_numpy()), name="Worst Steer", line=dict(color=C_RED, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(x=w_t, y=w_seg["alpha_front_pct"].to_numpy(), name="Worst Front%", line=dict(color=C_BLUE, dash="dash")), row=3, col=1)
        fig.add_trace(go.Scatter(x=w_t, y=w_seg["alpha_rear_pct"].to_numpy(), name="Worst Rear%", line=dict(color=C_YELLOW, dash="dash")), row=3, col=1)
        fig.add_trace(go.Scatter(x=w_t, y=w_seg["bilateral_score"].to_numpy(), name="Worst Bilateral", line=dict(color=C_RED, dash="dash")), row=4, col=1)

        # Shade bilateral-state window (>70) for best lap
        bscore = b_seg["bilateral_score"].to_numpy()
        state_active = bscore >= 70
        if state_active.any():
            t_on = b_t[state_active][0]
            t_off = b_t[state_active][-1]
            fig.add_vrect(x0=t_on, x1=t_off, fillcolor=C_OVERLAY, line_width=0, row=4, col=1)

        fig.update_yaxes(range=[0, 1.1], row=1, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=4, col=1)
        fig.update_layout(
            title=f"Track Corner {track_num + 1} — Lap {best_lap} (solid) vs Lap {worst_lap} (dashed)",
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font_color="#eee",
            height=700,
        )
        html += _div(fig)

        # Text summary box
        b_bil_dur = float(np.sum(bscore >= 70)) / tick_rate
        w_bil_dur = float(np.sum(w_seg["bilateral_score"].to_numpy() >= 70)) / tick_rate
        diff_peak = best_c.bilateral_peak_time - worst_c.bilateral_peak_time
        diff_throttle = (
            (best_c.throttle_app_time - worst_c.throttle_app_time)
            if not math.isnan(best_c.throttle_app_time) and not math.isnan(worst_c.throttle_app_time)
            else float("nan")
        )
        peak_dir = "EARLY" if diff_peak < 0 else "LATE"
        throttle_note = (
            f"Earlier throttle on best lap by {abs(diff_throttle):.2f}s"
            if not math.isnan(diff_throttle) and diff_throttle < -0.05
            else (
                f"Later throttle on best lap by {abs(diff_throttle):.2f}s"
                if not math.isnan(diff_throttle) and diff_throttle > 0.05
                else "Similar throttle timing"
            )
        )

        key_diff = "Brake release timing is late on worst lap." if diff_peak > 0.15 else "Bilateral timing is similar on both laps."
        if worst_c.overlap_duration < 0.1:
            key_diff = "Overlap window too brief on worst lap — brake releases after bilateral state closes."

        summary = f"""
<div class="summary-box">
<pre><strong>CORNER {track_num + 1} ANALYSIS</strong>
Best lap (Lap {best_c.lap}, bilateral {best_c.peak_bilateral:.0f}):
  Bilateral peak:       {best_c.bilateral_peak_time:.2f}s after corner entry
  Overlap duration:     {best_c.overlap_duration:.2f}s
  Bilateral state (≥70):{b_bil_dur:.2f}s
  Throttle application: {_fmt_time(best_c.throttle_app_time) if not math.isnan(best_c.throttle_app_time) else "—"}s after entry

Worst lap (Lap {worst_c.lap}, bilateral {worst_c.peak_bilateral:.0f}):
  Bilateral peak:       {worst_c.bilateral_peak_time:.2f}s after entry  ({abs(diff_peak):.2f}s {peak_dir})
  Overlap duration:     {worst_c.overlap_duration:.2f}s
  Bilateral state (≥70):{w_bil_dur:.2f}s
  Throttle application: {_fmt_time(worst_c.throttle_app_time) if not math.isnan(worst_c.throttle_app_time) else "—"}s after entry

KEY DIFFERENCE: {key_diff}
{throttle_note}
</pre>
</div>"""
        html += summary
        shown += 1

    return html


def _page_timing_scatter(
    corners: list[Corner],
) -> str:
    html = "<h2>Input Timing Analysis</h2>"
    if not corners:
        return html + "<p>No corner data.</p>"

    # Chart 1: Time-to-peak vs peak bilateral
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=[
                            "Time from Corner Entry to Bilateral Peak vs Peak Score",
                            "Overlap Duration vs Bilateral Score During Overlap",
                        ])

    peak_times = [c.bilateral_peak_time for c in corners if not math.isnan(c.bilateral_peak_time)]
    peak_scores = [c.peak_bilateral for c in corners if not math.isnan(c.bilateral_peak_time)]
    lap_labels = [f"Lap {c.lap}, Crnr {c.track_corner_num+1}" for c in corners if not math.isnan(c.bilateral_peak_time)]

    if peak_times:
        median_x = float(np.median(peak_times))
        median_y = float(np.median(peak_scores))

        fig.add_trace(go.Scatter(
            x=peak_times, y=peak_scores, mode="markers",
            marker=dict(color=[_color_bilateral(s) for s in peak_scores], size=8),
            text=lap_labels, hovertemplate="%{text}<br>Time: %{x:.2f}s<br>Score: %{y:.0f}<extra></extra>",
            name="Corners",
        ), row=1, col=1)
        fig.add_hline(y=median_y, line_dash="dash", line_color=C_GREY, annotation_text=f"Median {median_y:.0f}", row=1, col=1)
        fig.add_vline(x=median_x, line_dash="dash", line_color=C_GREY, annotation_text=f"Median {median_x:.2f}s", row=1, col=1)

    # Chart 2: Overlap duration vs bilateral during overlap
    ov_dur = [c.overlap_duration for c in corners if c.overlap_duration > 0.05]
    ov_bil = [c.bilateral_during_overlap for c in corners if c.overlap_duration > 0.05]
    ov_labels = [f"Lap {c.lap}, Crnr {c.track_corner_num+1}" for c in corners if c.overlap_duration > 0.05]

    if ov_dur:
        fig.add_trace(go.Scatter(
            x=ov_dur, y=ov_bil, mode="markers",
            marker=dict(color=[_color_bilateral(s) for s in ov_bil], size=8),
            text=ov_labels, hovertemplate="%{text}<br>Overlap: %{x:.2f}s<br>Bilateral: %{y:.0f}<extra></extra>",
            name="Overlap quality",
        ), row=2, col=1)

    fig.update_xaxes(title_text="Time to Bilateral Peak (s)", row=1, col=1)
    fig.update_yaxes(title_text="Peak Bilateral Score", range=[0, 100], row=1, col=1)
    fig.update_xaxes(title_text="Overlap Duration (s)", row=2, col=1)
    fig.update_yaxes(title_text="Bilateral Score During Overlap", range=[0, 100], row=2, col=1)
    fig.update_layout(
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font_color="#eee",
        height=700,
    )
    return html + _div(fig)


def _page_channel_heatmap(
    df: pd.DataFrame,
    corners: list[Corner],
) -> str:
    html = "<h2>Channel Coordination Heatmap</h2>"
    if not corners:
        return html + "<p>No corner data.</p>"

    track_corners = sorted(set(c.track_corner_num for c in corners))
    laps = sorted(set(c.lap for c in corners))

    phases = ["Entry (0-33%)", "Mid (33-67%)", "Exit (67-100%)"]

    # For each corner×lap×phase: compute dominant channel (1, 2, or 3)
    data_z: list[list[float]] = []  # rows = track corners, cols = phase×lap (not shown here)
    # Simpler: average dominant channel across laps for each track_corner × phase
    z_matrix = np.zeros((len(track_corners), 3))

    for ti, tcn in enumerate(track_corners):
        tc_list = [c for c in corners if c.track_corner_num == tcn]
        for pi in range(3):
            phase_channels = []
            for c in tc_list:
                seg = df.iloc[c.entry_idx:c.exit_idx + 1]
                n = len(seg)
                p0 = n * pi // 3
                p1 = n * (pi + 1) // 3
                if p1 <= p0:
                    continue
                sub = seg.iloc[p0:p1]
                c1 = sub["channel1"].abs().mean()
                c2 = sub["channel2"].mean()
                c3 = sub["channel3"].mean()
                dominant = max(1, 2, 3, key=lambda k: [c1, c2, c3][k - 1])
                phase_channels.append(dominant)
            z_matrix[ti, pi] = float(np.mean(phase_channels)) if phase_channels else 0.0

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=phases,
        y=[f"Corner {t + 1}" for t in track_corners],
        colorscale=[[0.0, C_RED], [0.5, C_YELLOW], [1.0, C_BLUE]],
        zmin=1, zmax=3,
        colorbar=dict(
            title="Dominant Channel",
            tickvals=[1, 2, 3],
            ticktext=["Ch1 Pull", "Ch2 Push", "Ch3 Feet"],
        ),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Dominant Input Channel per Corner Phase (averaged over all laps)",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font_color="#eee",
        height=max(300, len(track_corners) * 40 + 100),
    )
    return html + _div(fig)


def _page_scrub_analysis(
    df: pd.DataFrame,
    lap_summaries: list[dict],
    steering_ratio: float = 13.0,
) -> str:
    """Page 7: Steering angle vs lateral G — find front tyre scrub threshold per lap."""
    html = "<h2>Steering Scrub Analysis</h2>"
    html += (
        "<p>Lateral G vs road wheel angle (steering ratio {:.0f}:1). "
        "White line = theoretical linear front efficiency from peak G sample. "
        "Dashed red vertical = scrub threshold (angle of peak mean gLat — "
        "past this the front is past its peak slip angle).</p>"
    ).format(steering_ratio)

    RAD_TO_DEG = 57.2958
    BIN_WIDTH   = 0.5   # degrees

    lap_time_map = {r["lap"]: r.get("lap_time", float("nan")) for r in lap_summaries}
    any_chart = False

    for lap_num in sorted(df["Lap"].unique()):
        lap_mask = (
            (df["Lap"] == lap_num)
            & (df["Speed"] > 20)
            & (df["gLat"].abs() > 0.3)
        )
        seg = df[lap_mask]
        if len(seg) < 20:
            continue
        any_chart = True

        road_angle = np.abs(seg["SteeringWheelAngle"].to_numpy(float)) / steering_ratio * RAD_TO_DEG
        glat_abs   = np.abs(seg["gLat"].to_numpy(float))
        brake      = seg["Brake"].to_numpy(float)
        throttle   = seg["Throttle"].to_numpy(float)

        entry_mask = brake > 0.05
        exit_mask  = (~entry_mask) & (throttle > 0.1)
        mid_mask   = ~(entry_mask | exit_mask)

        # Reference line: slope through origin via the peak-gLat sample
        peak_idx    = int(np.argmax(glat_abs))
        peak_glat   = float(glat_abs[peak_idx])
        peak_angle  = float(road_angle[peak_idx])
        slope = peak_glat / peak_angle if peak_angle > 0 else 0.0

        angle_max = max(float(road_angle.max()), 1.0)
        ref_x = np.linspace(0.0, angle_max * 1.08, 120)
        ref_y = slope * ref_x if slope > 0 else None

        # Scrub threshold: 0.5° bins — find bin with highest mean gLat
        n_bins     = max(1, int(math.ceil(angle_max / BIN_WIDTH)))
        bin_edges  = np.array([i * BIN_WIDTH for i in range(n_bins + 1)])
        bin_centers: list[float] = []
        bin_means:   list[float] = []
        for i in range(n_bins):
            in_bin = (road_angle >= bin_edges[i]) & (road_angle < bin_edges[i + 1])
            if in_bin.sum() >= 3:
                bin_centers.append(bin_edges[i] + BIN_WIDTH / 2)
                bin_means.append(float(glat_abs[in_bin].mean()))

        scrub_angle: Optional[float] = None
        if len(bin_means) >= 2:
            scrub_angle = bin_centers[int(np.argmax(bin_means))]

        # Lap time for title
        lt = lap_time_map.get(int(lap_num), float("nan"))
        title_time = _fmt_time(lt) if not math.isnan(lt) else "—"

        fig = go.Figure()
        phase_cfg = [
            (entry_mask, "#3498db", "Entry (brake>5%)"),
            (mid_mask,   "#2ecc71", "Mid-corner"),
            (exit_mask,  "#e67e22", "Exit (throttle>10%)"),
        ]
        for mask, color, label in phase_cfg:
            if mask.any():
                fig.add_trace(go.Scatter(
                    x=road_angle[mask], y=glat_abs[mask],
                    mode="markers",
                    marker=dict(color=color, size=3, opacity=0.55),
                    name=label,
                ))

        if ref_y is not None:
            fig.add_trace(go.Scatter(
                x=ref_x, y=ref_y, mode="lines",
                line=dict(color="rgba(255,255,255,0.6)", dash="dash", width=1.5),
                name="Theoretical linear",
                hoverinfo="skip",
            ))

        if scrub_angle is not None:
            fig.add_vline(
                x=scrub_angle,
                line_dash="dash", line_color=C_RED,
                annotation_text=f"Scrub threshold: {scrub_angle:.1f}°",
                annotation_position="top right",
                annotation_font_color=C_RED,
            )

        fig.update_layout(
            title=f"Lap {lap_num}  ({title_time}) — Road Wheel Angle vs Lateral G",
            xaxis_title="Road Wheel Angle (degrees)",
            yaxis_title="Lateral G",
            xaxis_range=[0, None],
            yaxis_range=[0, None],
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#16213e",
            font_color="#eee",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        html += _div(fig)

    if not any_chart:
        html += "<p>Insufficient cornering data for scrub analysis.</p>"
    return html


def _page_micro_slide_map(
    df: pd.DataFrame,
    track_x: np.ndarray,
    track_y: np.ndarray,
    best_lap_num: int,
    corners: list[Corner],
    tick_rate: int = 60,
) -> str:
    """Page 8: Track map with rear-slip events and per-lap slide statistics."""
    html = "<h2>Micro-Slide Track Map</h2>"

    has_slip = "rear_slip" in df.columns and df["rear_slip"].abs().max() > 1e-6
    if not has_slip:
        html += (
            "<p>Wheel speed data not available in this session — "
            "enable telemetry variables in iRacing to see rear slip events.</p>"
        )

    CTRL_THRESH = 0.03
    SIGF_THRESH = 0.08
    MIN_SAMPLES = 3

    def _find_segs(slip: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
        segs: list[tuple[int, int]] = []
        above = slip > threshold
        i = 0
        while i < len(above):
            if above[i]:
                j = i + 1
                while j < len(above) and above[j]:
                    j += 1
                if j - i >= min_len:
                    segs.append((i, j))
                i = j
            else:
                i += 1
        return segs

    # ── Track map (best lap) ───────────────────────────────────────────────────
    lap_mask_np = (df["Lap"] == best_lap_num).to_numpy()
    x        = track_x[lap_mask_np]
    y        = track_y[lap_mask_np]
    rear_slip_l = (
        df.loc[df["Lap"] == best_lap_num, "rear_slip"].to_numpy(float)
        if has_slip else np.zeros(int(lap_mask_np.sum()))
    )

    ctrl_segs = _find_segs(rear_slip_l, CTRL_THRESH, MIN_SAMPLES)
    sigf_segs = _find_segs(rear_slip_l, SIGF_THRESH, 1)  # no min-duration for red
    sigf_set  = set()
    for s, e in sigf_segs:
        sigf_set.update(range(s, e))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=C_GREY, size=2, opacity=0.3),
        name="Track", hoverinfo="skip",
    ))

    if has_slip:
        orange_first = True
        for s, e in ctrl_segs:
            orange_idx = [i for i in range(s, e) if i not in sigf_set]
            if not orange_idx:
                continue
            fig.add_trace(go.Scatter(
                x=x[orange_idx], y=y[orange_idx], mode="markers",
                marker=dict(color="#e67e22", size=6),
                name="Controlled rear slip (3–8%)",
                legendgroup="controlled",
                showlegend=orange_first,
                hovertemplate="Controlled slip<extra></extra>",
            ))
            orange_first = False

        red_first = True
        for s, e in sigf_segs:
            seg_idx = list(range(s, e))
            fig.add_trace(go.Scatter(
                x=x[seg_idx], y=y[seg_idx], mode="markers",
                marker=dict(color=C_RED, size=7),
                name="Significant rear slide (>8%)",
                legendgroup="significant",
                showlegend=red_first,
                hovertemplate="Significant slide<extra></extra>",
            ))
            red_first = False

    fig.update_layout(
        title=f"Track Map — Rear Slip Events (Lap {best_lap_num})",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font_color="#eee",
        height=600,
        legend=dict(bgcolor="#1a1a2e", bordercolor="#3498db", borderwidth=1),
    )
    html += _div(fig)

    # ── Per-lap statistics table ───────────────────────────────────────────────
    def _peak_slip_section(lap_df: pd.DataFrame, lap_corners: list[Corner]) -> str:
        slip_arr = lap_df["rear_slip"].to_numpy(float) if has_slip else np.zeros(len(lap_df))
        dist_arr = lap_df["LapDistPct"].to_numpy(float)
        n_bins   = 20
        edges    = np.linspace(0.0, 1.0, n_bins + 1)
        counts   = [
            int(np.sum(slip_arr[(dist_arr >= edges[i]) & (dist_arr < edges[i + 1])] > CTRL_THRESH))
            for i in range(n_bins)
        ]
        if not any(c > 0 for c in counts):
            return "—"
        peak_bin   = int(np.argmax(counts))
        bin_center = float((edges[peak_bin] + edges[peak_bin + 1]) / 2)

        if not lap_corners:
            return f"{bin_center:.0%} of lap"

        best_corner: Optional[Corner] = None
        best_d = float("inf")
        for c in lap_corners:
            c_entry = df.iloc[c.entry_idx]["LapDistPct"]
            c_exit  = df.iloc[c.exit_idx]["LapDistPct"]
            c_ctr   = (float(c_entry) + float(c_exit)) / 2.0
            d = abs(bin_center - c_ctr)
            if d < best_d:
                best_d = d
                best_corner = c

        if best_corner is None:
            return f"{bin_center:.0%} of lap"

        c_entry  = float(df.iloc[best_corner.entry_idx]["LapDistPct"])
        c_exit   = float(df.iloc[best_corner.exit_idx]["LapDistPct"])
        span     = c_exit - c_entry
        if span > 0:
            rel   = (bin_center - c_entry) / span
            phase = "Corner entry" if rel < 0.33 else ("Mid-corner" if rel < 0.67 else "Corner exit")
        else:
            phase = "Mid-corner"
        return f"Corner {best_corner.track_corner_num + 1} — {phase}"

    rows = ""
    for lap_num in sorted(df["Lap"].unique()):
        lmask  = df["Lap"] == lap_num
        lap_df = df[lmask]
        slip   = lap_df["rear_slip"].to_numpy(float) if has_slip else np.zeros(len(lap_df))
        s_ctrl = float(np.sum(slip > CTRL_THRESH)) / tick_rate
        s_sigf = float(np.sum(slip > SIGF_THRESH)) / tick_rate
        lc     = [c for c in corners if c.lap == lap_num]
        peak   = _peak_slip_section(lap_df, lc)
        rows += (
            f"<tr>"
            f"<td>{lap_num}</td>"
            f"<td>{s_ctrl:.1f}</td>"
            f"<td>{s_sigf:.1f}</td>"
            f"<td style='text-align:left'>{peak}</td>"
            f"</tr>"
        )

    html += f"""
<h3 style="margin-top:24px">Per-Lap Rear Slip Statistics</h3>
<table class="data-table">
<thead><tr>
  <th>Lap</th>
  <th style="color:#e67e22">Rear Slip &gt;3% (s)</th>
  <th style="color:{C_RED}">Rear Slip &gt;8% (s)</th>
  <th style="text-align:left">Peak Slip Section</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>
<p style="color:#aaa;font-size:12px;margin-top:8px">
  Peak Slip Section = track position with most rear slip &gt;3%.
  Entry = trail brake issue &nbsp;|&nbsp; Mid = balance &nbsp;|&nbsp; Exit = throttle/differential
</p>"""

    return html


def _page_validation(validation: dict, df: pd.DataFrame) -> str:
    corr = validation.get("bilateral_glat_corr", 0)
    plausible = validation.get("plausible", False)
    sign_ok = validation.get("sign_ok", False)

    warnings = []
    if not sign_ok:
        warnings.append("⚠️  Front and rear slip angles have opposite signs at peak lateral G — check VelocityX sign convention.")
    if not plausible:
        warnings.append(f"⚠️  Max slip angles exceed 20°: front={validation.get('max_alpha_front_deg', 0):.1f}°, rear={validation.get('max_alpha_rear_deg', 0):.1f}°. Wheelbase or steering ratio may be wrong.")
    if corr < 0.3:
        warnings.append(f"⚠️  Low correlation of bilateral score with lateral G (r={corr:.2f}). Derivation may be unreliable for this session.")
    if not warnings:
        warnings.append(f"✅ Derivation looks plausible. Bilateral↔gLat correlation r={corr:.2f}.")

    warn_html = "".join(f"<p class='warn'>{w}</p>" for w in warnings)

    # Sample table from fastest corner
    peak_glat_idx = df["gLat"].abs().idxmax()
    start = max(0, peak_glat_idx - 50)
    end = min(len(df) - 1, peak_glat_idx + 50)
    sample = df.iloc[start:end:5]

    rows = "".join(
        f"<tr><td>{row['time']:.2f}</td><td>{row['VelocityX']:.2f}</td>"
        f"<td>{row['VelocityY']:.2f}</td><td>{row['YawRate']:.3f}</td>"
        f"<td>{row['alpha_front_deg']:.2f}</td><td>{row['alpha_rear_deg']:.2f}</td>"
        f"<td>{row['bilateral_score']:.1f}</td></tr>"
        for _, row in sample.iterrows()
    )

    return f"""
<h2>Validation</h2>
{warn_html}
<table class="data-table" style="font-size:12px">
<thead><tr>
  <th>Time</th><th>VelX (m/s)</th><th>VelY (m/s)</th><th>YawRate (r/s)</th>
  <th>alpha_front °</th><th>alpha_rear °</th><th>Bilateral</th>
</tr></thead>
<tbody>{rows}</tbody>
</table>"""


# ── Main entry point ──────────────────────────────────────────────────────────

CSS = """
body { margin:0; background:#0d1117; color:#cdd6f4; font-family: 'Segoe UI', sans-serif; }
.header { background:#1a1a2e; padding:20px 30px; border-bottom:2px solid #3498db; }
.header h1 { margin:0; font-size:24px; color:#eee; }
.header p  { margin:4px 0 0; color:#aaa; font-size:13px; }
.tabs { display:flex; gap:4px; padding:12px 20px; background:#16213e; flex-wrap:wrap; }
.tab-btn { background:#2c3e50; color:#aaa; border:none; padding:8px 18px; border-radius:6px;
           cursor:pointer; font-size:13px; transition:all .2s; }
.tab-btn:hover, .tab-btn.active { background:#3498db; color:#fff; }
.tab-content { display:none; padding:20px 30px; }
.tab-content.active { display:block; }
.data-table { border-collapse:collapse; width:100%; font-size:13px; }
.data-table th, .data-table td { border:1px solid #2c3e50; padding:6px 12px; text-align:right; }
.data-table thead { background:#1a1a2e; color:#3498db; }
.data-table tbody tr:hover { background:#1a1a2e; }
.summary-box { background:#1a1a2e; border-left:3px solid #3498db; margin:10px 0;
               padding:12px 18px; border-radius:4px; }
.summary-box pre { margin:0; font-size:12px; color:#aaa; font-family:monospace; white-space:pre-wrap; }
.warn { background:#e74c3c22; border-left:3px solid #e74c3c; padding:8px 12px; margin:6px 0; border-radius:3px; }
.what-box { background:#2ecc7122; border:1px solid #2ecc71; padding:14px 18px; margin:14px 0;
            border-radius:6px; font-size:13px; }
"""

JS = """
function showTab(id) {
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(e => e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector('[data-tab="' + id + '"]').classList.add('active');
}
document.addEventListener('DOMContentLoaded', () => showTab('overview'));
"""

WHAT_IS_GOOD = """
<div class="what-box">
<strong>Signs of bilateral simultaneous state:</strong> Both front and rear slip % above 70 simultaneously for >0.3 s.
Bilateral score above 70 during the overlap window. Small axle imbalance (&lt;15%) at apex.
Corner feels like <em>"the car is on rails"</em>.
<br><br>
<strong>Signs of sequential driving:</strong> Rear spikes first, then front catches up. Large imbalance throughout.
Overlap duration &lt;0.1 s. Bilateral score peaks at entry or exit, not sustained mid-corner.
<br><br>
<strong>Signs of understeer dominance:</strong> Front axle consistently higher than rear. Low bilateral score
even at high lateral G. Long trail brake with minimal rear slip contribution.
</div>
"""


def generate_report(
    df: pd.DataFrame,
    corners: list[Corner],
    lap_summaries: list[dict],
    track_x: np.ndarray,
    track_y: np.ndarray,
    validation: dict,
    output_path: str,
    session_label: str = "",
    tick_rate: int = 60,
    compare: Optional[dict] = None,
    steering_ratio: float = 13.0,
) -> str:
    best_lap = max(lap_summaries, key=lambda r: r["mean_bilateral"])["lap"] if lap_summaries else 1

    tab_defs = [
        ("overview",    "Session Overview"),
        ("trackmap",    "Track Map"),
        ("deep_dive",   "Corner Deep Dive"),
        ("best_worst",  "Best vs Worst Lap"),
        ("timing",      "Input Timing"),
        ("heatmap",     "Channel Heatmap"),
        ("validation",  "Validation"),
        ("scrub",       "Steering Scrub"),
        ("micro_slide", "Micro-Slide Map"),
    ]

    tab_buttons = "".join(
        f'<button class="tab-btn" data-tab="{tid}" onclick="showTab(\'{tid}\')">{label}</button>'
        for tid, label in tab_defs
    )

    pages = {
        "overview":    _page_overview(lap_summaries),
        "trackmap":    _page_trackmap(df, track_x, track_y, best_lap),
        "deep_dive":   _page_corner_deep_dive(df, corners),
        "best_worst":  _page_best_vs_worst(df, corners, lap_summaries, tick_rate),
        "timing":      _page_timing_scatter(corners),
        "heatmap":     _page_channel_heatmap(df, corners),
        "validation":  _page_validation(validation, df),
        "scrub":       _page_scrub_analysis(df, lap_summaries, steering_ratio),
        "micro_slide": _page_micro_slide_map(df, track_x, track_y, best_lap, corners, tick_rate),
    }

    tab_contents = "".join(
        f'<div id="{tid}" class="tab-content">{WHAT_IS_GOOD if tid == "overview" else ""}{content}</div>'
        for tid, content in pages.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Bilateral Balance Analysis — {session_label}</title>
<style>{CSS}</style>
</head>
<body>
<div class="header">
  <h1>Bilateral Balance Analysis</h1>
  <p>{session_label} &nbsp;|&nbsp; {len(corners)} corners detected &nbsp;|&nbsp; {len(lap_summaries)} laps</p>
</div>
<div class="tabs">{tab_buttons}</div>
{tab_contents}
<script>{JS}</script>
</body>
</html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
