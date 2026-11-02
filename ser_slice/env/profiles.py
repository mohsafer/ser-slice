"""Profile registry: named 5G/6G configurations for the environment."""
from ser_slice.config import make_config

PROFILES = ("5g", "6g")


def build(profile="5g", overrides=None):
    """Return a full config dict for `profile` with optional overrides."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    return make_config(profile, overrides)
