import json
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / 'tyre_audio_config.json'

DEFAULTS = {
    "input_device":     None,
    "output_device":    None,
    "engine_cut_db":    -20,
    "tyre_boost_db":    12,
    "notch_freqs":      [200, 320, 400, 600, 800],
    "notch_bw_pct":     15,
    "tyre_band_hz":     [800, 8000]
}

def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg = {**DEFAULTS, **json.load(f)}
    else:
        cfg = DEFAULTS.copy()
    return cfg

def save_config(cfg):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved to {CONFIG_PATH}")
