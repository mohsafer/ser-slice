# SER-Slice — Stochastic Entropy-Regularized DQN for 5G→6G RAN Slicing

A **fully theoretical, pure-software simulation project**: a Reinforcement Learning agent
learns to dynamically split radio resources of a 5G (and later 6G) base station across
network slices. No hardware, no SDRs, no testbeds — only stochastic network models, an
MDP, and a Deep Q-Network.

**Lineage.** The methodology is inherited from
[SER-DQN](https://github.com/mohsafer/serdqn), which applied a stochastic,
entropy-regularized Double DQN to TCP congestion control in ns-3. SER-Slice ports that
same algorithmic core — soft (entropy-regularized) value targets, Brownian diffusion in
the Bellman backup, and the four-arm ablation A1–A4 — from a wired bottleneck to the
5G→6G radio access network.

| SER-DQN (TCP, ns-3)              | SER-Slice (5G→6G RAN, pure Python)                       |
|----------------------------------|----------------------------------------------------------|
| ns-3 dumbbell, 3 flows           | Single gNB, 3 slices (eMBB / URLLC / mMTC)               |
| Bottleneck queue, 1.5 Mbps       | Per-slice queues over a shared PRB pool                  |
| Action: send-rate multiplier     | Action: PRB split across slices (+ RIS mode in 6G)       |
| State: cwnd, RTT, tput, loss     | State: backlogs, arrivals, SINR, delay, SLA violations   |
| Reward: tput − RTT − loss, clip  | Reward: slice utilities − SLA penalties, clip [−5, 2]    |
| Ablation A1–A4                   | Same four arms, identical semantics                      |
| eval harness → CSV, monitor → figs | Same discipline, same artifacts                        |

---

## 1. Research questions

- **RQ1 (learnability).** Can a DQN learn a PRB-splitting policy that beats static and
  heuristic schedulers on cross-slice KPIs under bursty, heterogeneous traffic?
- **RQ2 (stochastic targets).** Do the SER-DQN ingredients — entropy-regularized soft
  values and diffusion-perturbed backups — improve robustness over a vanilla DDQN in the
  *inherently stochastic* wireless environment (fading, shadowing, random arrivals)?
  This is the exact A1–A4 question SER-DQN asked for TCP, re-asked for the RAN.
- **RQ3 (5G→6G transfer).** When the network evolves from a 5G profile to a 6G profile
  (wider carriers, RIS, ISAC sensing, stronger non-stationarity), does the learned
  controller adapt or transfer? What does AI-native control need to survive the
  generation shift?

## 2. Goals and non-goals

**Goals**
1. A self-contained, reproducible, **zero-hardware** network simulator with mathematically
   explicit 3GPP-style models (pathloss, shadowing, block fading, queueing, SLA accounting).
2. A Gymnasium-API-compatible `RanSlicingEnv` with a fully specified MDP.
3. A reference implementation of the SER-DQN family (arms A1–A4) plus non-RL baselines.
4. A serdqn-style evaluation pipeline: JSONL training logs, CSV evaluation traces,
   paper-grade figures, seed-replicated statistics.

**Non-goals (by design)**
- Bit-accurate PHY/MAC (no LDPC, no HARQ timing, no real SCS slot math beyond capacity).
- Multi-cell interference coordination (single cell with a stochastic interference model).
- Hardware-in-the-loop, OTA measurements, real gNB traces.
- Anything requiring an installation: the reference core runs on the **Python standard
  library only**; numpy/torch are an optional acceleration path (§10).

## 3. System model

### 3.1 Topology and numerology

One gNB serving K = 3 slices. Each slice i has U_i = 8 modeled UEs (channel realizations);
mMTC additionally has N_dev = 100 sporadic devices at the traffic layer.

| Parameter        | 5G profile | 6G profile   | Notes                              |
|------------------|-----------:|-------------:|------------------------------------|
| Carrier f_c      | 3.5 GHz    | 28 GHz       | mmWave/sub-THz flavor              |
| Bandwidth        | 20 MHz     | 100 MHz      |                                    |
| PRBs             | 106        | 273          | Δf_PRB = 12 × SCS = 360 kHz        |
| Decision interval| 10 ms      | 10 ms        | one "TTI bundle" per RL step       |
| Episode          | 1000 steps = 10 s simulated                          |    |
| Pathloss model   | 3GPP TR 38.901 UMa-NLOS                              |    |

### 3.2 Channel model

Per UE u, per decision step t:

```
PL_u(t)  = 13.54 + 39.08·log10(d_u) + 20·log10(f_c)          [dB]   (38.901 UMa-NLOS)
ξ_u(t)   = ρ·ξ_u(t−1) + sqrt(1−ρ²)·σ_sh·N(0,1)               [dB]   (AR(1) shadowing)
h_u(t)   ~ Rayleigh, |h|² ~ Exp(1)                            (block fading, i.i.d. per step)
N_u      = −174 + 10·log10(Δf_PRB) + NF                       [dBm]  (thermal noise per PRB)
I(t)     = N_u / load(t)                                             (stochastic inter-cell interference)
SINR_u(t) = P_tx − PL_u − ξ_u + 10·log10(|h_u|²) − 10·log10(I + N)   [dB]
```

with σ_sh = 6 dB, ρ = 0.9, P_tx = 43 dBm, NF = 7 dB. Per-slice achievable spectral
efficiency (uniform PRB assumption):

```
η_i(t) = η_impl · E_u[ log2(1 + SINR_u(t)) ],    η_impl = 0.75
```

Serving r PRBs to slice i for dt yields service rate R_i = r · Δf_PRB · η_i bits/s.
The 6G profile adds (§6): RIS gain (−G_RIS dB effective pathloss) and random blockage
events (multiplicative SINR crash), both gated by configuration.

### 3.3 Traffic models (3GPP-flavored)

| Slice | Model                                    | Defaults                                        | SLA                        |
|-------|------------------------------------------|-------------------------------------------------|----------------------------|
| eMBB  | Poisson arrivals, LogNormal packet sizes | λ = 3 pkt/s, mean 0.5 MB, σ = 0.8               | throughput (best effort)   |
| URLLC | Poisson arrivals, small packets          | λ = 50 pkt/s, 80–200 B, deadline 10 ms (6G: 5)  | deadline violation rate    |
| mMTC  | 100 devices × Bernoulli per step         | p = 0.02/step, 20–200 B                         | mean access delay          |

**Non-stationarity.** The 5G profile applies a mild sinusoidal load swing (±15%).
The 6G profile switches the traffic regime at fixed fractions of the episode
(e.g., ×0.6 → ×1.6 → ×1.0), emulating the fast load dynamics an AI-native RAN is
expected to track — and making RQ3 a real adaptation test.

### 3.4 Queues and SLA accounting

Each slice owns a FIFO queue of packets (bits + arrival time + optional deadline):
- **Overflow:** drops when backlog exceeds B_max,i (per-slice, configurable).
- **Serving:** bits-granular FIFO at rate R_i·dt; sojourn time recorded per packet.
- **URLLC deadline:** a packet served after its deadline counts as a violation;
  expired-at-head packets are dropped and counted.
- **mMTC access delay:** EWMA of packet sojourn (α = 0.05).

### 3.5 Scheduling abstraction

The agent replaces the PRB *allocator* each decision step. Within a slice, service is
FIFO — the RL problem is the cross-slice split, which is where slicing policy actually
lives. Non-RL baselines (§7) plug into the same interface.

## 4. MDP formulation

### 4.1 State (default: 22 dims, 25 in the 6G profile)

Per slice (6 × 3 = 18, all normalized):

```
[ backlog/B_max , arrival EWMA/arr_scale , SINR/30 dB ,
  sojourn-or-age/deadline-scale , violation EWMA , served EWMA/served_scale ]
```

plus `[ previous PRB-split fractions (3) , episode phase (1) ]`.

**6G extras (+3):** ISAC interference forecast (AR prediction of next-step load + sensor
noise), current regime indicator (sensing-informed load bucket), RIS active flag.

### 4.2 Action

Discrete index into a **PRB-split simplex grid**: the PRB pool is partitioned into
`units` blocks of `prbs_per_unit` (5G: 10 × 10 PRBs; 6G: 10 × 27 PRBs), and one action is
an integer split of the 10 units across the 3 slices → **66 actions**
(|a₁+a₂+a₃ = 10|). Grid size is profile- and config-invariant, which keeps the action
space identical across generations — deliberately, so RQ3 measures environment transfer,
not interface transfer.

**6G extension:** the action is the product `split × RIS-mode` with two RIS
configurations → 132 actions. Switching RIS mode costs a one-step transient
(all SINRs × 0.5), creating a genuine persist-vs-reconfigure trade-off.

### 4.3 Reward

Per step, clipped to [−5, 2] exactly as in SER-DQN:

```
R_t = w_e·ũ_e + w_u·ũ_u + w_m·ũ_m
      − w_v·violation_rate − w_o·overflow_rate − w_i·idle_PRB_fraction
```

- ũ_e = log1p(eMBB throughput in Mbps) / log1p(50) — concave utility
- ũ_u = URLLC in-deadline success ratio this window
- ũ_m = 1 − clip(mMTC access delay / 1 s, 0, 1)
- default weights w_e=1.0, w_u=0.5, w_m=0.5, w_v=2.0, w_o=0.5, w_i=0.1 —
  chosen so that near-perfect service saturates the top of the [−5, 2] window
  while starvation, violations, and overflow pull the reward down, keeping the
  whole learning signal inside the clip range

The asymmetry (URLLC weighted above eMBB, violations penalized hard) encodes 5G QoS
class priorities; weight sweeps are part of the experiment plan (§8).

### 4.4 Transition and termination

Environment transitions are the composition of traffic arrivals → channel → queue
service → accounting; they are stochastic and *partially observed* (the agent sees
normalizations of the above, not the raw generator state). Episodes truncate (never
terminate early) after `episode_steps`; `terminated=False, truncated=True` at the end,
so bootstrapping remains active — matching a continuing-operation RAN.

## 5. Agent: SER-DQN in the RAN

### 5.1 Architecture

MLP [obs → 64 → 64 → |A|] with ReLU, replay buffer 50k, batch 64, Adam lr 5e-4,
γ = 0.99, ε-greedy 1.0 → 0.05 over 20k steps, target network synced every 500 steps,
Huber loss. (Reference skeleton: hidden 32, batch 8 — see §10.)

### 5.2 The SER target (the inherited core)

Let Q_θ (online) select, Q_θ⁻ (target) evaluate, as in Double DQN:

```
a*        = argmax_a Q_θ(s′, a)
q*(s′)    = Q_θ⁻(s′, a*)
```

**Soft (entropy-regularized) value** at temperature α:

```
V_soft(s′) = α · logsumexp_a( Q_θ⁻(s′, a) / α )
```

**Multiplicative soft-diffusion.** Draw ΔW ~ N(0,1) and form the backup

```
y = r + γ·(1−done) · [ V(s′) + ( q*(s′) − V(s′) ) · ( 1 + g·|ΔW| ) ]
```

The DDQN gain (q* − V) multiplies the stochastic factor, so the target variance scales
with the magnitude of the advantage over the soft value — noisy value estimates are
treated as distributions, not exact numbers, which is precisely the failure mode a
fading, bursty environment stresses.

### 5.3 Ablation arms (semantics identical to SER-DQN)

| Arm | Name        | Value V(s′)              | Diffusion g |
|-----|-------------|--------------------------|-------------|
| A1  | vanilla DDQN| hard max                 | 0           |
| A2  | soft-only   | V_soft                   | 0           |
| A3  | **full SER**| V_soft                   | g = σ > 0   |
| A4  | gain-only   | hard max                 | g = σ > 0   |

All arms share seeds, networks, buffers, and ε-schedules; **only the target rule
differs**. That isolation is what makes RQ2 answerable.

## 6. 5G → 6G evolution profiles

One simulator, two configurations — the "generation" is a profile, not a second codebase:

| Aspect            | 5G profile            | 6G profile                              |
|-------------------|-----------------------|-----------------------------------------|
| Carrier / PRBs    | 3.5 GHz / 106         | 28 GHz / 273                            |
| URLLC deadline    | 10 ms                 | 5 ms                                    |
| Load dynamics     | mild sinusoid         | regime switches mid-episode             |
| RIS               | —                     | 2 modes, reconfig transient             |
| ISAC sensing      | —                     | interference forecast + regime in state |
| Blockage          | —                     | Bernoulli outage events (RIS mitigates) |

**Protocol for RQ3:** (i) train on 5G, evaluate on 6G (zero-shot transfer);
(ii) fine-tune on 6G (adaptation); (iii) train from scratch on 6G (upper bound).
Report the transfer gap. The RIS/ISAC knobs can each be toggled to attribute which 6G
feature breaks (or helps) the learned policy.

## 7. Baselines and evaluation protocol

**Baselines** (same env, same grid): `static` fixed split (50/30/20), `rr` rotating
dominant slice, `prop` backlog-proportional split, and (later) PF per-slice scheduler.
An `oracle-greedy` upper reference may be added for diagnostics.

**Protocol.** 20 training seeds per arm/config; 100 greedy evaluation episodes per
checkpoint; mean ± 95% t-interval; Welch's t-test for A3 vs A1 (the RQ2 headline).
Every artifact is seed-tagged: `runs/<id>/{train.jsonl, eval.csv, ckpt.json, config.json}`.

**Metrics:** per-slice throughput; URLLC deadline violation rate; URLLC delay CDF
(p50/p99); mMTC access delay; PRB utilization; spectral efficiency; Jain fairness across
slice utilities; learning-curve AUC.

**Figures (paper-grade, produced by `monitor.py`):**
F1 learning curves per arm (reward EWMA ± CI) · F2 KPI bars vs baselines ·
F3 URLLC delay CDF · F4 target-variance vs |advantage| scatter (stochasticity analysis,
serdqn's signature diagnostic) · F5 5G↔6G transfer bars · F6 PRB-split heatmap over an
episode (policy interpretability).

## 8. Experiment plan

| # | Experiment | Question | Outputs |
|---|------------|----------|---------|
| E1 | Agent vs baselines, 5G profile | RQ1 | F2, KPI table |
| E2 | A1–A4 ablation, 5G + 6G | RQ2 | F1, F3, F4, Welch tests |
| E3 | 5G→6G zero-shot / fine-tune / scratch | RQ3 | F5 |
| E4 | Robustness: traffic-shift, weight sweeps, seed variance | generality | tables, F6 |

## 9. Repository architecture

```
ser_slice/
├── config.py            JSON config loading + deep merge (base ← profile ← CLI)
├── env/
│   ├── shim.py          Gymnasium-API-compatible Env/Box/Discrete (stdlib fallback)
│   ├── channel.py       38.901 pathloss, AR(1) shadowing, block fading, SINR
│   ├── traffic.py       eMBB/URLLC/mMTC generators, regime/non-stationarity engine
│   ├── queues.py        FIFO queues, deadline expiry, SLA accounting
│   ├── profiles.py      profile registry: "5g", "6g"
│   └── ran_env.py       RanSlicingEnv: MDP glue, action grid, reward, KPIs
├── agents/
│   ├── nets.py          pure-Python MLP + manual backprop + Adam (reference impl)
│   ├── dqn_agent.py     SER-DQN: replay, ε-schedule, SER target, arms A1–A4
│   └── baselines.py     static / rr / prop
├── train.py             training loop → runs/<id>/train.jsonl + ckpt.json
├── eval_harness.py      greedy evaluation → eval.csv + summary
├── monitor.py           figures F1–F6 (matplotlib, optional)
└── utils.py             seeding, EWMA, Poisson sampler, JSONL writer
```

**Reference implementation note.** `agents/nets.py` implements the network and manual
backpropagation in pure Python so the entire MDP + SER-DQN machinery is runnable,
testable, and gradcheck-able with **zero dependencies** — the math is the artifact. For
full-scale experiments (§10, M2+), swap the backend to numpy/torch behind the same
interface; the environment needs no change.

## 10. Implementation stages

| Stage | Content | Status |
|-------|---------|--------|
| M0 | This design + stdlib reference skeleton (env, SER-DQN, arms, baselines, harness, tests) | **this deliverable** |
| M1 | Env hardening: per-UE scheduling detail, correlated shadowing maps, config sweeps | next |
| M2 | numpy/torch backend, full hyperparameters, E1 + E2 campaigns | |
| M3 | 6G profile completion (RIS/ISAC/blockage), E3 | |
| M4 | Statistical analysis, figures F1–F6, results write-up | |
| M5 | Extensions: PPO/SAC comparison, MARL (per-slice agents), ns-3/LENA port path | |

**Acceleration path.** `requirements.txt` lists optional extras (numpy, torch,
gymnasium, matplotlib, pandas). When installed: gymnasium provides the real `Env` base,
torch replaces the reference MLP (same SER math), numpy vectorizes channel/traffic,
matplotlib enables `monitor.py`. None are required for the skeleton to run.

## 11. Limitations

- Shannon-level capacity with uniform intra-slice PRB assumption: preserves the
  *allocation* trade-off, ignores per-UE scheduling subtleties inside a slice (M1).
- Interference is a stochastic scalar process, not a geometric multi-cell simulation.
- Pure-Python reference is intentionally small (hidden 32/64): it validates the math and
  the pipeline, not wall-clock-scale training — that is torch's job in M2.
- DQN's discrete grid quantizes the split; a continuous-action head (SAC) is the natural
  M5 extension.

## 12. References

1. mohsafer, *SER-DQN: Stochastic Entropy-Regularized DQN for TCP Congestion Control*
   (official code), https://github.com/mohsafer/serdqn
2. 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*
3. 3GPP TR 38.913, *Study on scenarios and requirements for next generation access*;
   ITU-R M.2160 (IMT-2030 framework) for 6G usage scenarios
4. Mnih et al., *Human-level control through deep RL* (DQN), Nature 2015;
   van Hasselt et al., *Double DQN*, AAAI 2016
5. Haarnoja et al., *Soft Actor-Critic* (entropy-regularized values), ICML 2018
6. Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
