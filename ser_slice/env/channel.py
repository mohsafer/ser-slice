"""Radio channel: 3GPP TR 38.901 UMa-NLOS pathloss, AR(1) shadowing,
Rayleigh block fading, stochastic inter-cell interference, and the 6G-profile
extras (RIS gain, blockage events). Pure stdlib."""
import math
import random

# TR 38.901 Table 7.4.1-1, UMa NLOS: PL = 13.54 + 39.08 log10(d3D) + 20 log10(fc)
UMA_NLOS_A, UMA_NLOS_B, UMA_NLOS_C = 13.54, 39.08, 20.0


class ChannelModel:
    """Per-UE channel realizations for one gNB with K slices.

    Each slice has `ues_per_slice[i]` UE channel realizations; the slice's
    achievable spectral efficiency is the UE-mean of log2(1+SINR), scaled by
    the implementation loss. Fading is i.i.d. per decision step (block
    fading); shadowing evolves as AR(1); interference load as AR(1).
    """

    def __init__(self, cfg, rng: random.Random):
        e = cfg["env"]
        self.rng = rng
        self.prb_bw_hz = e["prb_bw_hz"]
        self.p_tx_dbm = e["p_tx_dbm"]
        self.nf_db = e["nf_db"]
        self.noise_psd_dbm_hz = e["noise_psd_dbm_hz"]
        self.fc_ghz = e["fc_ghz"]
        self.dist_min, self.dist_max = e["dist_min_m"], e["dist_max_m"]
        self.sigma_sh = e["sigma_shadow_db"]
        self.rho_sh = e["shadow_rho"]
        self.impl_loss = e["impl_loss"]
        self.ues_per_slice = list(e["ues_per_slice"])
        self.i_rho = e["interference_rho"]
        self.load_target = e["load_target"]
        self.load_std = e["load_noise_std"]
        self.p_block = e["p_block"]
        self.blockage_factor = e["blockage_sinr_factor"]
        self.blockage_ris_factor = e["blockage_ris_factor"]
        self.ris_gain_db = e["ris_gain_db"]
        self.distances = None
        self.shadow = None
        self.load = self.load_target
        self.blocked = False

    def new_episode(self):
        self.distances = [
            [self.rng.uniform(self.dist_min, self.dist_max) for _ in range(u)]
            for u in self.ues_per_slice
        ]
        self.shadow = [[0.0] * u for u in self.ues_per_slice]
        self.load = self.load_target
        self.blocked = False

    def _pathloss_db(self, d_m):
        return UMA_NLOS_A + UMA_NLOS_B * math.log10(max(d_m, 1.0)) + UMA_NLOS_C * math.log10(self.fc_ghz)

    def _noise_dbm(self):
        return self.noise_psd_dbm_hz + 10.0 * math.log10(self.prb_bw_hz) + self.nf_db

    def step(self, ris_active):
        """Advance one decision interval.

        Returns (effs, mean_sinr_db, blocked): per-slice achievable spectral
        efficiency [bps/Hz], per-slice mean SINR [dB], and whether a blockage
        event hit this step.
        """
        noise_dbm = self._noise_dbm()
        # Inter-cell interference: AR(1) around the target load; lighter
        # neighboring load -> less interference.
        self.load = self.i_rho * self.load + (1.0 - self.i_rho) * self.load_target
        self.load = max(0.05, min(1.0, self.load + self.rng.gauss(0.0, self.load_std)))
        i_dbm = noise_dbm - 10.0 * math.log10(self.load)

        self.blocked = self.p_block > 0.0 and self.rng.random() < self.p_block
        effs, mean_sinr_db = [], []
        for i, dists in enumerate(self.distances):
            ue_sinr_lin = []
            for u in range(len(dists)):
                # AR(1) shadowing in dB (loss term)
                xi = self.rho_sh * self.shadow[i][u] + math.sqrt(1.0 - self.rho_sh ** 2) * self.sigma_sh * self.rng.gauss(0.0, 1.0)
                self.shadow[i][u] = xi
                h2 = -math.log(1.0 - self.rng.random() * 0.999999)  # |h|^2 ~ Exp(1)
                pl = self._pathloss_db(dists[u])
                if ris_active:
                    pl -= self.ris_gain_db
                rx_dbm = self.p_tx_dbm - pl - xi + 10.0 * math.log10(h2)
                sinr_db = rx_dbm - 10.0 * math.log10(10 ** (i_dbm / 10.0) + 10 ** (noise_dbm / 10.0))
                sinr_lin = 10.0 ** (sinr_db / 10.0)
                if self.blocked:
                    sinr_lin *= self.blockage_ris_factor if ris_active else self.blockage_factor
                ue_sinr_lin.append(sinr_lin)
            eff = self.impl_loss * sum(math.log2(1.0 + s) for s in ue_sinr_lin) / len(ue_sinr_lin)
            effs.append(eff)
            mean_sinr_db.append(10.0 * math.log10(sum(ue_sinr_lin) / len(ue_sinr_lin) + 1e-12))
        self.last_sinr_db = mean_sinr_db
        return effs, mean_sinr_db, self.blocked

    def forecast_load(self):
        """ISAC-style one-step-ahead load prediction (deterministic AR mean +
        sensor noise). Only exposed when the profile enables ISAC."""
        pred = self.i_rho * self.load + (1.0 - self.i_rho) * self.load_target
        return pred
