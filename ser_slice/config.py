"""JSON config loading and deep merge: base <- profile <- CLI overrides.

Configs live in `<repo>/configs/`: `base.json` holds the full 5G defaults,
`profile_6g.json` holds only the 6G deltas (see DESIGN.md section 6).
"""
from pathlib import Path

from ser_slice.utils import deep_merge, load_json

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
PROFILES = {"5g": "profile_5g.json", "6g": "profile_6g.json"}


def make_config(profile="5g", overrides=None, base_path=None):
    """Build a full config dict for a named profile, with optional overrides."""
    cfg = load_json(Path(base_path) if base_path else CONFIG_DIR / "base.json")
    if profile:
        cfg = deep_merge(cfg, load_json(CONFIG_DIR / PROFILES[profile]))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg.setdefault("meta", {})["profile"] = profile
    return cfg
