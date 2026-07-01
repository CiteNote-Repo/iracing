#!/usr/bin/env python3
"""
Bilateral Axle Balance Analyzer for iRacing telemetry.

Usage:
  python main.py --latest
  python main.py --file session.ibt
  python main.py --file a.ibt --compare b.ibt --label1 "Dry" --label2 "Wet"
  python main.py --demo
  python main.py --file session.ibt --wheelbase 2.65 --steering-ratio 13.5
"""

import argparse
import os
import sys
import time
import webbrowser
from pathlib import Path

import numpy as np

# ── Add project directory to path ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, get_car_params
from ibt_reader import read_ibt, get_latest_ibt
from slip_calculator import compute_slip_angles, compute_validation
from corner_detector import detect_corners, get_lap_times
from channel_analyzer import compute_channels
from bilateral_scorer import compute_bilateral, compute_absolute_utilization, score_corners, score_laps, build_track_xy
from tyre_energy_tracker import compute_tyre_energy
from report_generator import generate_report


def _process(
    df,
    car_params: dict,
    cfg: dict,
    is_demo: bool = False,
) -> tuple:
    tick_rate: int = df.attrs.get("tick_rate", 60)

    # Slip angles (skip derivation if demo pre-computed values are already set)
    if is_demo and "alpha_front_pct" in df.columns:
        pass
    else:
        df = compute_slip_angles(
            df,
            wheelbase_m=car_params["wheelbase_m"],
            cg_to_front_ratio=car_params["cg_to_front_ratio"],
            steering_ratio=car_params["steering_ratio"],
        )

    df = compute_channels(df, tick_rate=tick_rate,
                          brake_threshold=cfg["overlap_brake_threshold"],
                          steer_threshold=cfg["overlap_steer_threshold"])
    df = compute_bilateral(df)
    df = compute_absolute_utilization(df, peak_glat_g=cfg.get("peak_glat_g", 1.65))
    df = compute_tyre_energy(df)

    df, corners, track_positions = detect_corners(
        df,
        corner_threshold_g=cfg["corner_threshold_g"],
        tick_rate=tick_rate,
    )

    corners = score_corners(df, corners, tick_rate=tick_rate,
                            bilateral_threshold=cfg["bilateral_threshold_pct"])

    lap_times = get_lap_times(df)
    lap_summaries = score_laps(df, corners, lap_times,
                               bilateral_threshold=cfg["bilateral_threshold_pct"])

    track_x, track_y = build_track_xy(df, tick_rate=tick_rate)

    validation = compute_validation(df)

    return df, corners, lap_summaries, track_x, track_y, validation


def _run_analysis(args: argparse.Namespace, cfg: dict) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Demo mode ──────────────────────────────────────────────────────────────
    if args.demo:
        print("Generating synthetic demo telemetry …")
        t0 = time.time()
        from demo_generator import generate_demo_data, apply_demo_slip_angles

        df = generate_demo_data()
        df = apply_demo_slip_angles(df)

        df.attrs.update({"tick_rate": 60, "record_count": len(df), "filepath": "demo", "car_name": "Demo Car"})
        car_params = get_car_params(cfg, "")

        df, corners, lap_summaries, track_x, track_y, validation = _process(
            df, car_params, cfg, is_demo=True
        )

        # Override lap labels for demo
        styles = {1: "Bilateral", 2: "Oversteer", 3: "Understeer", 4: "Sequential"}
        for s in lap_summaries:
            s["label"] = styles.get(s["lap"], "")

        out_path = str(output_dir / "bilateral_demo.html")
        generate_report(
            df, corners, lap_summaries, track_x, track_y, validation,
            output_path=out_path,
            session_label="DEMO — four synthetic driving styles",
            tick_rate=60,
            steering_ratio=car_params["steering_ratio"],
        )
        elapsed = time.time() - t0
        print(f"Demo report generated in {elapsed:.1f}s → {out_path}")
        if not args.no_browser:
            webbrowser.open(f"file://{os.path.abspath(out_path)}")
        return

    # ── Real IBT file ──────────────────────────────────────────────────────────
    filepath = args.file
    if args.latest:
        filepath = get_latest_ibt(cfg["iracing_telemetry_dir"])
        if not filepath:
            print(f"No IBT files found in {cfg['iracing_telemetry_dir']}")
            sys.exit(1)
        print(f"Using latest IBT: {filepath}")

    if not filepath:
        print("Specify --file, --latest, or --demo")
        sys.exit(1)
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"Reading {filepath} …")
    t0 = time.time()
    df = read_ibt(filepath)
    record_count = df.attrs["record_count"]
    tick_rate = df.attrs["tick_rate"]
    print(f"  {record_count} frames at {tick_rate} Hz ({record_count / tick_rate:.0f}s)")

    # CLI overrides take precedence over config
    car_params = get_car_params(cfg, df.attrs.get("car_name", ""))
    if args.wheelbase:
        car_params["wheelbase_m"] = args.wheelbase
    if args.steering_ratio:
        car_params["steering_ratio"] = args.steering_ratio

    print(f"  Wheelbase: {car_params['wheelbase_m']}m  Steering ratio: {car_params['steering_ratio']}:1")

    df, corners, lap_summaries, track_x, track_y, validation = _process(df, car_params, cfg)
    print(f"  {len(corners)} corners in {len(lap_summaries)} laps")

    label = Path(filepath).stem
    out_path = str(output_dir / f"bilateral_{label}.html")

    if args.compare:
        if not Path(args.compare).exists():
            print(f"Compare file not found: {args.compare}")
            sys.exit(1)
        print(f"Reading compare file {args.compare} …")
        df2 = read_ibt(args.compare)
        car_params2 = get_car_params(cfg, df2.attrs.get("car_name", ""))
        df2, corners2, lap_summaries2, tx2, ty2, val2 = _process(df2, car_params2, cfg)
        compare_data = dict(df=df2, corners=corners2, lap_summaries=lap_summaries2,
                            track_x=tx2, track_y=ty2, validation=val2,
                            label=args.label2 or Path(args.compare).stem)
        label1 = args.label1 or label
    else:
        compare_data = None
        label1 = args.label1 or label

    generate_report(
        df, corners, lap_summaries, track_x, track_y, validation,
        output_path=out_path,
        session_label=label1,
        tick_rate=tick_rate,
        compare=compare_data,
        steering_ratio=car_params["steering_ratio"],
    )

    elapsed = time.time() - t0
    corr = validation.get("bilateral_glat_corr", 0)
    plausible = validation.get("plausible", False)

    print(f"\nReport: {out_path}  ({elapsed:.1f}s)")
    print(f"Validation: bilateral↔gLat r={corr:.2f}  plausible={'YES' if plausible else 'NO'}")
    if corr < 0.3:
        print("  ⚠  Low correlation — check steering ratio or wheelbase settings")

    if not args.no_browser:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bilateral Axle Balance Analyzer for iRacing IBT files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--file", metavar="PATH", help="Path to IBT file")
    source.add_argument("--latest", action="store_true", help="Auto-find latest IBT in telemetry dir")
    source.add_argument("--demo", action="store_true", help="Generate demo report from synthetic data")

    parser.add_argument("--compare", metavar="PATH", help="Second IBT file to compare against")
    parser.add_argument("--label1", metavar="TEXT", help="Label for primary session")
    parser.add_argument("--label2", metavar="TEXT", help="Label for comparison session")
    parser.add_argument("--wheelbase", type=float, metavar="M", help="Override wheelbase (metres)")
    parser.add_argument("--steering-ratio", type=float, metavar="N", help="Override steering ratio")
    parser.add_argument("--output-dir", metavar="PATH", help="Output directory (default: current dir)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser after generating report")

    args = parser.parse_args()

    if not any([args.file, args.latest, args.demo]):
        parser.print_help()
        sys.exit(0)

    cfg = load_config()
    _run_analysis(args, cfg)


if __name__ == "__main__":
    main()
