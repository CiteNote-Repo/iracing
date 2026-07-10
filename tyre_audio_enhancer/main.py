#!/usr/bin/env python3
import argparse
import sys

import sounddevice as sd

from config import load_config, save_config


def list_devices():
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    print("-" * 60)
    for i, d in enumerate(devices):
        direction = []
        if d['max_input_channels'] > 0:
            direction.append("IN")
        if d['max_output_channels'] > 0:
            direction.append("OUT")
        tag = "/".join(direction)
        print(f"  [{i:2d}] [{tag:6s}] {d['name']}")
    print()


def parse_device(device_str):
    """Return int if numeric string, otherwise return as string name."""
    if device_str is None:
        return None
    try:
        return int(device_str)
    except (ValueError, TypeError):
        return device_str


def find_device(name_fragment, kind):
    """Return device index matching name_fragment for the given kind."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        channel_key = f'max_{kind}_channels'
        if d.get(channel_key, 0) > 0 and name_fragment.lower() in d['name'].lower():
            return d['name']
    return None


def auto_detect_input():
    """Try to find VB-Cable output device (the loopback capture side)."""
    for fragment in ["CABLE Output", "VB-Audio", "VB-Cable"]:
        result = find_device(fragment, "input")
        if result:
            return result
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Tyre Audio Enhancer — amplify tyre sounds, attenuate engine"
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="List all audio devices and exit")
    parser.add_argument("--input", metavar="DEVICE",
                        help="Input device name (default: auto-detect VB-Cable)")
    parser.add_argument("--output", metavar="DEVICE",
                        help="Output device name (default: system default)")
    parser.add_argument("--engine-cut", type=float, metavar="DB",
                        help="Engine attenuation in dB (default: -20)")
    parser.add_argument("--tyre-boost", type=float, metavar="DB",
                        help="Tyre frequency boost in dB (default: 12)")
    parser.add_argument("--notch-freqs", type=int, nargs="+", metavar="HZ",
                        help="Engine harmonic frequencies to notch out")
    parser.add_argument("--calibrate", action="store_true",
                        help="10-second listen mode to auto-detect engine harmonics")
    parser.add_argument("--save-preset", action="store_true",
                        help="Save current settings to config file")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        sys.exit(0)

    cfg = load_config()

    # CLI args override config
    if args.input:
        cfg["input_device"] = parse_device(args.input)
    if args.output:
        cfg["output_device"] = parse_device(args.output)
    if args.engine_cut is not None:
        cfg["engine_cut_db"] = args.engine_cut
    if args.tyre_boost is not None:
        cfg["tyre_boost_db"] = args.tyre_boost
    if args.notch_freqs:
        cfg["notch_freqs"] = args.notch_freqs

    # Auto-detect input if not specified
    if cfg["input_device"] is None:
        detected = auto_detect_input()
        if detected:
            print(f"Auto-detected input: {detected}")
            cfg["input_device"] = detected
        else:
            print("Could not auto-detect VB-Cable. Use --input DEVICE or --list-devices.")
            sys.exit(1)

    if args.calibrate:
        from calibrate import calibrate
        notch_freqs = calibrate(cfg["input_device"])
        cfg["notch_freqs"] = notch_freqs
        print("\nRun again without --calibrate to start the enhancer with these frequencies.")
        if args.save_preset:
            save_config(cfg)
        sys.exit(0)

    if args.save_preset:
        save_config(cfg)

    from enhancer import TyreAudioEnhancer
    enhancer = TyreAudioEnhancer(
        input_device=cfg["input_device"],
        output_device=cfg["output_device"],
        engine_cut_db=cfg["engine_cut_db"],
        tyre_boost_db=cfg["tyre_boost_db"],
        notch_freqs=cfg["notch_freqs"],
    )
    try:
        enhancer.run()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
