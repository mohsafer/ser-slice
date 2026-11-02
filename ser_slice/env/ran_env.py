"""RanSlicingEnv: the 5G/6G RAN-slicing MDP (Gymnasium-API compatible).

One gNB, K slices sharing a PRB pool. Each decision step (one "TTI bundle",
default 10 ms) the agent picks a PRB split (and, in the 6G profile, a RIS
mode). The environment composes traffic arrivals -> channel -> queue service
-> SLA accounting and returns the next observation and reward.

State / action / reward are specified in DESIGN.md section 4; the PRB-split
action grid is the ordered compositions of `action_units` over the slices
(default 66 actions), optionally multiplied by RIS modes in the 6G profile.
"""
import math
import random

from ser_slice.env.shim import Box, Discrete, Env
from ser_slice.env.channel import ChannelModel
from ser_slice.env.queues import SliceQueue
from ser_slice.env.traffic import TrafficEngine
from ser_slice.utils import clip, ewma

SLICES = ("eMBB", "URLLC", "mMTC")
URLLC_IDX = 1


def build_action_grid(units, n_slices):
    """All ordered splits of `units` into `n_slices` non-negative parts."""
    if n_slices == 1:
        return [(units,)]
    out = []
    for first in range(units + 1):
        for rest in build_action_grid(units - first, n_slices - 1):
            out.append((first,) + rest)
    return out


class RanSlicingEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.rng = random.Random(seed)
        e = cfg["env"]
        self.K = e["n_slices"]
        self.dt = e["decision_dt_s"]
        self.total_prbs = e["total_prbs"]
        self.prb_bw = e["prb_bw_hz"]
        self.prbs_per_unit = e["prbs_per_unit"]
        self.episode_steps = e["episode_steps"]
        self.b_max = list(e["b_max_bits"])
        self.ris_modes = e["ris_modes"]
        self.ris_gain_db = e["ris_gain_db"]
        self.ris_transient = e["ris_switch_transient"]
        self.isac = e["isac"]
        self.isac_std = e["isac_sensor_std"]

        self.grid = build_action_grid(e["action_units"], self.K)
        self.action_space = Discrete(len(self.grid) * self.ris_modes)

        self._base_obs = 6 * self.K + self.K + 1
        self.obs_size = self._base_obs + (3 if self.isac else 0)
        self.observation_space = Box([0.0] * self.obs_size, [2.0] * self.obs_size, (self.obs_size,))

        self.channel = ChannelModel(cfg, self.rng)
        self.traffic = TrafficEngine(cfg, self.rng)
        self.queues = [SliceQueue(b) for b in self.b_max]
        self.t = 0
        self.ris_mode = 0
        self.prev_fracs = [1.0 / self.K] * self.K
        self.arr_ewma = [None] * self.K
        self.served_ewma = [None] * self.K
        self.viol_ewma = None
        self.delay_ewma = [None] * self.K

    # ------------------------------------------------------------------ API

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.t = 0
        self.ris_mode = 0
        self.prev_fracs = [1.0 / self.K] * self.K
        self.arr_ewma = [None] * self.K
        self.served_ewma = [None] * self.K
        self.viol_ewma = None
        self.delay_ewma = [None] * self.K
        self.queues = [SliceQueue(b) for b in self.b_max]
        self.channel.new_episode()
        self.traffic.new_episode(self.episode_steps)
        return self._observe(), self._info({}, 0.0, 0.0, False)

    def step(self, action):
        e = self.cfg["env"]
        r = self.cfg["reward"]
        action = int(action)
        if not 0 <= action < self.action_space.n:
            raise ValueError(f"action {action} outside action space")

        split_idx, ris_idx = divmod(action, self.ris_modes)
        units = self.grid[split_idx]
        alloc_prbs = [u * self.prbs_per_unit for u in units]
        fracs = [u / float(e["action_units"]) for u in units]

        # RIS reconfiguration: switching costs a one-step transient of degraded
        # reflections (persist-vs-reconfigure trade-off, 6G profile only).
        transient = 1.0
        ris_active = False
        if self.ris_modes > 1:
            if ris_idx != self.ris_mode:
                transient = self.ris_transient
                self.ris_mode = ris_idx
            ris_active = self.ris_mode == 1 and transient == 1.0

        effs, sinr_dbs, blocked = self.channel.step(ris_active)
        effs = [eff * transient for eff in effs]

        now = self.t * self.dt
        now_end = now + self.dt
        mult = self.traffic.regime_multiplier(self.t)

        arrived = self.traffic.arrivals(now, mult)
        arrived_bits = [sum(p.bits for p in a) for a in arrived]
        overflow_bits = [0.0] * self.K
        for i in range(self.K):
            for p in arrived[i]:
                if not self.queues[i].push(p):
                    overflow_bits[i] += p.bits

        served_bits, served_pkts, sojourn_sum = [0.0] * self.K, [0] * self.K, [0.0] * self.K
        for i in range(self.K):
            rate = alloc_prbs[i] * self.prb_bw * effs[i]
            served_bits[i], served_pkts[i], sojourn_sum[i] = self.queues[i].serve(rate, self.dt, now_end)
        expired = self.queues[URLLC_IDX].drain_deadlines(now_end)

        # ------------------------------------------------------------- KPIs
        tput_bps = [b / self.dt for b in served_bits]
        viol_rate = expired / max(1, len(arrived[URLLC_IDX]))
        overflow_frac = sum(overflow_bits) / max(1e-9, sum(arrived_bits))
        idle_frac = 1.0 - sum(alloc_prbs) / float(self.total_prbs)
        mean_sojourn = [sojourn_sum[i] / served_pkts[i] if served_pkts[i] else None for i in range(self.K)]

        # ---------------------------------------------------------- EWMA state
        for i in range(self.K):
            self.arr_ewma[i] = ewma(self.arr_ewma[i], arrived_bits[i], 0.1)
            self.served_ewma[i] = ewma(self.served_ewma[i], served_bits[i], 0.1)
            if mean_sojourn[i] is not None:
                self.delay_ewma[i] = ewma(self.delay_ewma[i], mean_sojourn[i], 0.05)
        self.viol_ewma = ewma(self.viol_ewma, viol_rate, 0.05)
        self.prev_fracs = fracs

        # ------------------------------------------------------------ reward
        e_tput_mbps = tput_bps[0] / 1e6
        u_e = math.log1p(e_tput_mbps) / math.log1p(r["e_tput_norm_mbps"])
        u_u = 1.0 - viol_rate
        m_delay = self.delay_ewma[2] if self.delay_ewma[2] is not None else 0.0
        u_m = 1.0 - clip(m_delay / r["m_delay_norm_s"], 0.0, 1.0)
        reward = (
            r["w_e"] * u_e
            + r["w_u"] * u_u
            + r["w_m"] * u_m
            - r["w_v"] * viol_rate
            - r["w_o"] * overflow_frac
            - r["w_i"] * idle_frac
        )
        reward = clip(reward, r["clip"][0], r["clip"][1])

        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_steps
        obs = self._observe()

        kpi = {
            "thr_e_mbps": e_tput_mbps,
            "thr_u_mbps": tput_bps[1] / 1e6,
            "thr_m_mbps": tput_bps[2] / 1e6,
            "urllc_violation_rate": viol_rate,
            "mmtc_delay_s": m_delay,
            "overflow_frac": overflow_frac,
            "prb_util": 1.0 - idle_frac,
            "backlog_e_norm": self.queues[0].backlog / self.b_max[0],
            "sinr_e_db": sinr_dbs[0],
            "sinr_u_db": sinr_dbs[1],
            "sinr_m_db": sinr_dbs[2],
            "blocked": float(blocked),
            "regime_mult": mult,
            "u_e": u_e, "u_u": u_u, "u_m": u_m,
        }
        return obs, reward, terminated, truncated, self._info(kpi, reward, viol_rate, blocked)

    # ------------------------------------------------------------- helpers

    def nearest_action(self, fracs):
        """Grid index whose normalized split is L2-closest to `fracs`."""
        best, best_d = 0, None
        for idx, units in enumerate(self.grid):
            total = float(sum(units))
            d = sum((u / total - f) ** 2 for u, f in zip(units, fracs))
            if best_d is None or d < best_d:
                best, best_d = idx, d
        return best * self.ris_modes  # RIS mode 0 for baselines

    def _observe(self):
        obs = []
        now = self.t * self.dt
        sinr = getattr(self.channel, "last_sinr_db", None)
        for i in range(self.K):
            obs.append(self.queues[i].backlog / self.b_max[i])
            obs.append(clip((self.arr_ewma[i] or 0.0) / self.cfg["env"]["arr_scale_bits"][i], 0.0, 2.0))
            obs.append(clip((sinr[i] if sinr else 0.0) / self.cfg["env"]["sinr_norm_db"], 0.0, 2.0))
            if i == URLLC_IDX:
                delay = self.queues[i].hol_age(now) / self.cfg["env"]["delay_scale_s"][i]
            else:
                delay = (self.delay_ewma[i] or 0.0) / self.cfg["env"]["delay_scale_s"][i]
            obs.append(clip(delay, 0.0, 2.0))
            obs.append((self.viol_ewma or 0.0) if i == URLLC_IDX else 0.0)
            obs.append(clip((self.served_ewma[i] or 0.0) / self.cfg["env"]["served_scale_bits"][i], 0.0, 2.0))
        obs.extend(self.prev_fracs)
        obs.append(self.t / float(self.episode_steps))
        if self.isac:
            forecast = self.channel.forecast_load() + self.rng.gauss(0.0, self.isac_std)
            mult = self.traffic.regime_multiplier(self.t)
            obs.append(clip(forecast, 0.0, 2.0))
            obs.append(clip(mult / 2.0, 0.0, 2.0))
            obs.append(1.0 if (self.ris_modes > 1 and self.ris_mode == 1) else 0.0)
        return obs

    def _info(self, kpi, reward, viol_rate, blocked):
        return {
            "kpi": kpi,
            "reward": reward,
            "backlogs": [q.backlog for q in self.queues],
            "backlog_fracs": [q.backlog / self.b_max[i] for i, q in enumerate(self.queues)],
            "ris_mode": self.ris_mode,
            "regime_mult": self.traffic.regime_multiplier(self.t),
        }
