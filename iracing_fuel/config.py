import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "fuel_config.json")

DEFAULTS = {
    "safety_margin_pct": 10,
    "rolling_window_laps": 5,
    "overlay_position": {"x": 1700, "y": 50},
    "overlay_alpha": 0.85,
    "min_laps_before_auto": 2,
    "extra_buffer_laps": 0.5,
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULTS)
        return dict(DEFAULTS)
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)
    # Fill in any missing keys from defaults
    for k, v in DEFAULTS.items():
        if k not in data:
            data[k] = v
    return data


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def update_overlay_position(cfg: dict, x: int, y: int) -> None:
    cfg["overlay_position"]["x"] = x
    cfg["overlay_position"]["y"] = y
    save_config(cfg)
