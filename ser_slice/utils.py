"""Small stdlib-only helpers shared across SER-Slice."""
import json
import math
import random


def set_seed(seed: int) -> None:
    random.seed(seed)


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def ewma(prev, x, alpha):
    """Exponential moving average; `alpha` is the weight on the new sample."""
    return x if prev is None else alpha * x + (1.0 - alpha) * prev


def poisson(rng, lam):
    """Knuth's algorithm; adequate for the small per-step rates used here."""
    if lam <= 0.0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        p *= rng.random()
        if p <= L or k > 1000:  # safety valve for pathological lam
            return k
        k += 1


class LinearSchedule:
    """Linearly anneal from `start` to `end` over `decay_steps` calls."""

    def __init__(self, start, end, decay_steps):
        self.start, self.end = start, end
        self.decay_steps = max(1, decay_steps)
        self.t = 0

    def value(self):
        frac = min(1.0, self.t / self.decay_steps)
        return self.start + frac * (self.end - self.start)

    def step(self):
        self.t += 1


class JsonlWriter:
    """Append-only JSONL logger (serdqn-style training logs)."""

    def __init__(self, path):
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, record):
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def deep_merge(base, override):
    """New dict with `override` winning; nested dicts merge recursively."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
