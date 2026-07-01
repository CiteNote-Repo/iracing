import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "live_grip_config.json")

DEFAULTS = {
    "peak_glat_g": 2.30,
    "overlay_position": {"x": 1700, "y": 250},
    "overlay_alpha": 0.85,
    "audio_enabled": True,
    "audio_volume": 0.15,
    "min_speed_for_audio": 5.0,
    "car_overrides": {
        "Ferrari 296 GT3": {"peak_glat_g": 2.30, "steering_ratio": 13.0},
        "Acura NSX GT3 EVO22": {"peak_glat_g": 1.93, "steering_ratio": 14.0},
    },
}


def load_config() -> dict:
    if not os.path.exists(CONFIG_PATH):
        save_config(DEFAULTS)
        return dict(DEFAULTS)
    with open(CONFIG_PATH) as f:
        data = json.load(f)
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


def resolve_peak_glat(cfg: dict, car_name: str = "") -> float:
    overrides = cfg.get("car_overrides", {})
    for name, vals in overrides.items():
        if name.lower() in car_name.lower() or car_name.lower() in name.lower():
            return float(vals.get("peak_glat_g", cfg["peak_glat_g"]))
    return float(cfg["peak_glat_g"])


def resolve_steering_ratio(cfg: dict, car_name: str = "") -> float:
    overrides = cfg.get("car_overrides", {})
    for name, vals in overrides.items():
        if name.lower() in car_name.lower() or car_name.lower() in name.lower():
            return float(vals.get("steering_ratio", 13.0))
    return 13.0
