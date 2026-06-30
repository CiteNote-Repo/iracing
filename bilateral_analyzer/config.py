import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "iracing_telemetry_dir": str(Path.home() / "Documents/iRacing/telemetry"),
    "wheelbase_m": 2.7,
    "steering_ratio": 14.0,
    "cg_to_front_ratio": 0.45,
    "corner_threshold_g": 0.3,
    "bilateral_threshold_pct": 70,
    "overlap_brake_threshold": 0.05,
    "overlap_steer_threshold": 0.15,
    "peak_glat_g": 1.65,
    "car_overrides": {
        "Ferrari 296 GT3": {"wheelbase_m": 2.60, "steering_ratio": 13.0},
        "Lamborghini Huracan GT3": {"wheelbase_m": 2.62, "steering_ratio": 14.0},
        "Porsche 911 GT3 R": {"wheelbase_m": 2.46, "steering_ratio": 15.0},
        "Acura NSX GT3": {"wheelbase_m": 2.63, "steering_ratio": 14.5},
    },
}

CONFIG_FILE = Path(__file__).parent / "bilateral_config.json"


def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            user = json.load(f)
        cfg = {**DEFAULT_CONFIG, **user}
        # merge car_overrides dicts
        cfg["car_overrides"] = {
            **DEFAULT_CONFIG["car_overrides"],
            **user.get("car_overrides", {}),
        }
    else:
        cfg = DEFAULT_CONFIG.copy()
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"Created default config at {CONFIG_FILE}")
    return cfg


def _compact(s: str) -> str:
    """Lowercase, strip spaces/hyphens/underscores for fuzzy car name matching."""
    return s.lower().replace(" ", "").replace("-", "").replace("_", "")


def get_car_params(cfg: dict[str, Any], car_name: str) -> dict[str, Any]:
    params = {
        "wheelbase_m": cfg["wheelbase_m"],
        "steering_ratio": cfg["steering_ratio"],
        "cg_to_front_ratio": cfg["cg_to_front_ratio"],
    }
    name_lower   = car_name.lower()
    name_compact = _compact(car_name)
    for key, overrides in cfg.get("car_overrides", {}).items():
        key_lower   = key.lower()
        key_compact = _compact(key)
        # Match if either name contains the other, or compact forms match
        if (key_lower in name_lower
                or name_lower in key_lower
                or key_compact == name_compact
                or key_compact in name_compact
                or name_compact in key_compact):
            params.update(overrides)
            break
    return params
