"""Non-RL baseline policies on the same PRB-split action grid.

They observe the environment's `info` (backlogs) and return a grid index via
`RanSlicingEnv.nearest_action`, so they compete under exactly the same
action-space constraints as the DQN (DESIGN.md section 7)."""


class BasePolicy:
    def reset(self):
        pass

    def act(self, obs, info, env, greedy=True):
        raise NotImplementedError


class StaticSplit(BasePolicy):
    """Fixed split (default 50/30/20), rounded to the nearest grid point."""

    def __init__(self, fracs=(0.5, 0.3, 0.2)):
        self.fracs = list(fracs)

    def act(self, obs, info, env, greedy=True):
        return env.nearest_action(self.fracs)


class RoundRobin(BasePolicy):
    """Cycles the dominant slice each decision step."""

    def __init__(self, n_slices=3):
        self.n_slices = n_slices
        self.k = 0

    def reset(self):
        self.k = 0

    def act(self, obs, info, env, greedy=True):
        fracs = [0.0] * self.n_slices
        fracs[self.k % self.n_slices] = 1.0
        self.k += 1
        return env.nearest_action(fracs)


class BacklogProportional(BasePolicy):
    """Split proportional to normalized backlog; falls back to uniform."""

    def act(self, obs, info, env, greedy=True):
        fracs = info["backlog_fracs"]
        total = sum(fracs)
        if total <= 1e-9:
            fracs = [1.0 / len(fracs)] * len(fracs)
        else:
            fracs = [f / total for f in fracs]
        return env.nearest_action(fracs)


class SLAGuard(BasePolicy):
    """Heuristic: keep the previous split, but re-route toward URLLC when its
    deadline-violation EWMA (observed via backlog growth) rises. A simple
    greedy-SLA reference for the RL agent."""

    def __init__(self, n_slices=3, boost=0.15):
        self.n_slices = n_slices
        self.boost = boost
        self.prev = None

    def reset(self):
        self.prev = None

    def act(self, obs, info, env, greedy=True):
        if self.prev is None:
            fracs = [0.5, 0.3, 0.2][: self.n_slices]
        else:
            fracs = list(self.prev)
            # shift a small share toward URLLC when its backlog is non-empty
            if info["backlog_fracs"][1] > 0.05:
                fracs[1] = min(0.8, fracs[1] + self.boost)
                rest = (1.0 - fracs[1]) * (1.0 - fracs[1])
                fracs[0] = (1.0 - fracs[1]) * 0.7
                fracs[2] = (1.0 - fracs[1]) * 0.3
        self.prev = fracs
        return env.nearest_action(fracs)


BASELINES = {
    "static": StaticSplit,
    "rr": RoundRobin,
    "prop": BacklogProportional,
    "sla": SLAGuard,
}


def make_baseline(name, cfg=None):
    K = cfg["env"]["n_slices"] if cfg else 3
    if name == "static":
        return StaticSplit()
    if name == "rr":
        return RoundRobin(n_slices=K)
    if name == "prop":
        return BacklogProportional()
    if name == "sla":
        return SLAGuard(n_slices=K)
    raise ValueError(f"unknown baseline {name!r}; expected one of {sorted(BASELINES)}")
