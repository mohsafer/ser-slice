"""Traffic generators for the three slice classes plus the load-regime engine.

eMBB:  Poisson arrivals, LogNormal packet sizes (bursty large transfers).
URLLC: Poisson arrivals, small packets with an absolute deadline.
mMTC:  N devices each activating with probability p per step (sporadic tiny
       packets).

Non-stationarity: the 5G profile uses a mild sinusoidal load swing; the 6G
profile switches between piecewise-constant regime multipliers at fixed
fractions of the episode (DESIGN.md section 6)."""
import math
import random

from ser_slice.utils import poisson


class Packet:
    __slots__ = ("bits", "remaining", "t_arr", "deadline")

    def __init__(self, bits, t_arr, deadline=None):
        self.bits = bits
        self.remaining = bits
        self.t_arr = t_arr
        self.deadline = deadline


class TrafficEngine:
    def __init__(self, cfg, rng: random.Random):
        t = cfg["traffic"]
        self.rng = rng
        self.dt = cfg["env"]["decision_dt_s"]
        self.embb_rate = t["embb_rate_pps"]
        self.embb_mu = math.log(t["embb_size_mean_bits"]) - 0.5 * t["embb_size_sigma"] ** 2
        self.embb_sigma = t["embb_size_sigma"]
        self.urllc_rate = t["urllc_rate_pps"]
        self.urllc_lo, self.urllc_hi = t["urllc_size_bits"]
        self.urllc_deadline = t["urllc_deadline_s"]
        self.mmtc_devices = t["mmtc_devices"]
        self.mmtc_p = t["mmtc_p_per_step"]
        self.mmtc_lo, self.mmtc_hi = t["mmtc_size_bits"]
        self.regime_mode = t["regime_mode"]
        self.regime_period = t["regime_period_s"]
        self.regime_amp = t["regime_amplitude"]
        self.regime_multipliers = list(t["regime_multipliers"])
        self.regime_fractions = list(t["regime_switch_fractions"])
        self._switch_steps = []

    def new_episode(self, episode_steps):
        self._switch_steps = [int(f * episode_steps) for f in self.regime_fractions]

    def regime_multiplier(self, t_step):
        if self.regime_mode == "switches":
            idx = sum(1 for s in self._switch_steps if t_step >= s)
            idx = min(idx, len(self.regime_multipliers) - 1)
            return self.regime_multipliers[idx]
        # sinusoid (default 5G profile)
        return 1.0 + self.regime_amp * math.sin(2.0 * math.pi * t_step * self.dt / self.regime_period)

    def arrivals(self, now, mult):
        """Draw this interval's arrivals; returns a list (per slice) of Packet."""
        lam_e = self.embb_rate * self.dt * mult
        lam_u = self.urllc_rate * self.dt * mult
        p_m = min(0.5, self.mmtc_p * mult)

        embb = [
            Packet(math.exp(self.rng.gauss(self.embb_mu, self.embb_sigma)), now)
            for _ in range(poisson(self.rng, lam_e))
        ]
        urllc = [
            Packet(self.rng.uniform(self.urllc_lo, self.urllc_hi), now, now + self.urllc_deadline)
            for _ in range(poisson(self.rng, lam_u))
        ]
        n_m = sum(1 for _ in range(self.mmtc_devices) if self.rng.random() < p_m)
        mmtc = [
            Packet(self.rng.uniform(self.mmtc_lo, self.mmtc_hi), now)
            for _ in range(n_m)
        ]
        return [embb, urllc, mmtc]
