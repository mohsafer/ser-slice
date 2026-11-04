# SER-Slice — Stochastic Entropy-Regularized DQN for 5G→6G RAN Slicing [under development]

A **fully theoretical, pure-software** simulation project: a reinforcement learning
agent learns to dynamically split the radio resources of a 5G (and 6G) base station
across network slices (eMBB / URLLC / mMTC). No hardware, no SDRs, no testbed — only
3GPP-style stochastic models, an MDP, and a Deep Q-Network.

The methodology is inherited from [SER-DQN](https://github.com/mohsafer/serdqn)
(stochastic entropy-regularized Double DQN for TCP congestion control in ns-3): the
same soft-value targets, Brownian diffusion backups, and four-arm ablation (A1–A4),
re-asked for the 5G→6G radio access network. Full specification in
[DESIGN.md](DESIGN.md).

## Quickstart

Python 3.9+ with **zero dependencies** — the reference core is standard library only:

```bash
# 1. run the test suite (also pytest-compatible)
python3 tests/run_all.py

# 2. train the full SER arm (A3) on the 5G profile
python3 -m ser_slice.train --profile 5g --algo serdqn --arm A3 --episodes 20 --out runs/A3

# 3. train the ablation arms and a baseline
python3 -m ser_slice.train --profile 5g --algo serdqn --arm A1 --out runs/A1
python3 -m ser_slice.train --profile 5g --algo static --out runs/static

# 4. greedy evaluation -> CSV traces + summary (5G-trained ckpt on the 6G profile = transfer test)
python3 -m ser_slice.eval_harness --ckpt runs/A3/ckpt.json --profile 5g --episodes 100 --out runs/eval_A3_5g
python3 -m ser_slice.eval_harness --ckpt runs/A3/ckpt.json --profile 6g --episodes 100 --out runs/eval_A3_6g

# 5. figures (needs matplotlib, optional)
python3 -m ser_slice.monitor --curve runs/A3/train.jsonl runs/A1/train.jsonl \
    --labels A3 A1 --evalcsv runs/eval_A3_5g/eval.csv runs/eval_A3_6g/eval.csv --out reports/
```

Small-by-default note: the reference MLP is pure Python (hidden [32, 32]) — it
validates the math and the pipeline, not wall-clock-scale training. For full
campaigns, install the optional stack in `requirements.txt` (numpy/torch/gymnasium)
per `DESIGN.md` §10 and swap the agent backend behind the same interface.

## What's in the box

| Path | Role |
|------|------|
| [DESIGN.md](DESIGN.md) | Full design: system model, MDP, SER-DQN target math, 5G/6G profiles, experiments, roadmap |
| `configs/` | `base.json` (= 5G profile) + `profile_6g.json` deltas |
| `ser_slice/env/` | `RanSlicingEnv` (Gymnasium-API), channel (38.901 UMa-NLOS, AR(1) shadowing, block fading), traffic (eMBB/URLLC/mMTC + regime engine), queues, profiles |
| `ser_slice/agents/` | SER-DQN (arms A1–A4, pure-Python MLP + manual backprop + Adam), baselines (static/rr/prop/sla) |
| `ser_slice/train.py` `eval_harness.py` `monitor.py` | Training loop → `train.jsonl` + `ckpt.json`; greedy eval → `eval.csv` + `summary.json`; figures F1–F3 |
| `tests/` | 17 tests: env contract, gradient check, arm semantics, end-to-end pipeline (`python3 tests/run_all.py`) |

## The research questions

1. **Learnability** — can a DQN beat static/heuristic PRB-splitting under bursty,
   heterogeneous, fading traffic? (E1)
2. **Stochastic targets** — do entropy-regularized soft values + diffusion backups
   (the serdqn ingredients) improve robustness over vanilla DDQN in a stochastic RAN?
   Arms A1–A4 isolate exactly this. (E2)
3. **5G→6G transfer** — same simulator, two profiles: 6G adds wider carriers, RIS
   reconfiguration, ISAC sensing inputs, blockage, and regime-switching traffic. Train
   on 5G, deploy on 6G; measure the transfer gap. (E3)

## Status

- **M0 (this repo):** design + runnable reference skeleton — env, SER-DQN (A1–A4),
  baselines, training/eval/monitor pipeline, 17/17 tests passing.
- **Next (M1–M2):** env hardening, numpy/torch backend, full hyperparameters, E1+E2.
- See `DESIGN.md` §10 for the full roadmap.
