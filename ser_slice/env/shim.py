"""Gymnasium-compatible minimal API, with a zero-dependency fallback.

If gymnasium is installed, its real `Env`/`Box`/`Discrete` are re-exported so
`RanSlicingEnv` interops with gymnasium wrappers and checkers. Otherwise a
tiny faithful shim is provided so the project runs on the standard library
alone (DESIGN.md section 10: optional acceleration path).
"""
try:  # pragma: no cover - exercised only when gymnasium is available
    from gymnasium import Env
    from gymnasium.spaces import Box, Discrete
except ImportError:  # stdlib fallback

    class Env:
        """Minimal subset of the gymnasium 0.29 Env API used by this project."""

        metadata = {"render_modes": []}

        def reset(self, *, seed=None, options=None):
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def render(self):
            pass

        def close(self):
            pass

    class Box:
        def __init__(self, low, high, shape, dtype=None):
            self.low = list(low)
            self.high = list(high)
            self.shape = tuple(shape)

        def contains(self, x):
            if len(x) != self.shape[0]:
                return False
            return all(lo <= v <= hi for v, lo, hi in zip(x, self.low, self.high))

    class Discrete:
        def __init__(self, n):
            self.n = int(n)

        def contains(self, x):
            return isinstance(x, int) and 0 <= x < self.n
