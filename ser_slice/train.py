"""Training entry point.

    python -m ser_slice.train --profile 6g --algo serdqn --arm A3 --episodes 50

Writes runs/<out>/ : train.jsonl (per-episode records, serdqn-style),
config.json, and ckpt.json (final agent weights).
"""
import argparse
import os

from ser_slice.agents.baselines import make_baseline
from ser_slice.agents.dqn_agent import SERDQNAgent
from ser_slice.config import make_config
from ser_slice.env.profiles import PROFILES
from ser_slice.env.ran_env import RanSlicingEnv
from ser_slice.utils import JsonlWriter, ewma, save_json, set_seed

KPI_KEYS = (
    "thr_e_mbps",
    "thr_u_mbps",
    "thr_m_mbps",
    "urllc_violation_rate",
    "mmtc_delay_s",
    "overflow_frac",
    "prb_util",
)


def run(cfg, algo="serdqn", arm="A3", episodes=None, seed=0, out_dir="runs/default",
        max_steps=None, verbose=True):
    """Train `episodes` episodes; returns the final episode record."""
    episodes = episodes if episodes is not None else cfg["train"]["episodes"]
    set_seed(seed)
    env = RanSlicingEnv(cfg, seed=seed)
    agent = None
    policy = None
    if algo == "serdqn":
        agent = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=seed + 1, arm=arm)
    else:
        policy = make_baseline(algo, cfg)

    os.makedirs(out_dir, exist_ok=True)
    log = JsonlWriter(os.path.join(out_dir, "train.jsonl"))
    save_json(cfg, os.path.join(out_dir, "config.json"))
    if agent is not None:
        save_json({"algo": algo, "arm": arm, "seed": seed},
                  os.path.join(out_dir, "meta.json"))

    ret_ewma = None
    last = None
    for ep in range(episodes):
        obs, info = env.reset(seed=seed * 1000 + ep)
        if policy is not None:
            policy.reset()
        ep_ret, n = 0.0, 0
        kpi_sum = {k: 0.0 for k in KPI_KEYS}
        for _ in range(env.episode_steps if max_steps is None else min(max_steps, env.episode_steps)):
            a = agent.act(obs) if agent is not None else policy.act(obs, info, env)
            obs2, r, terminated, truncated, info = env.step(a)
            if agent is not None:
                agent.store(obs, a, r, obs2, terminated or truncated)
                agent.learn()
            obs = obs2
            ep_ret += r
            n += 1
            for k in KPI_KEYS:
                kpi_sum[k] += info["kpi"].get(k, 0.0)
            if terminated or truncated:
                break
        record = {"episode": ep, "return": ep_ret / max(1, n), "steps": n,
                  "algo": algo, "arm": arm, "seed": seed}
        ret_ewma = ewma(ret_ewma, record["return"], cfg["train"]["log_ewma_alpha"])
        record["return_ewma"] = ret_ewma
        if agent is not None:
            record["eps"] = agent.eps.value()
        for k in KPI_KEYS:
            record[k] = kpi_sum[k] / max(1, n)
        log.write(record)
        if verbose:
            kpi_str = " ".join(f"{k}={record[k]:.3f}" for k in KPI_KEYS[:4])
            print(f"[{algo}{'/' + arm if arm else ''} ep {ep:4d}] "
                  f"ret/step={record['return']:+.3f} ewma={record['return_ewma']:+.3f} {kpi_str}")
        last = record

    if agent is not None:
        save_json(agent.save_state(), os.path.join(out_dir, "ckpt.json"))
    log.close()
    return last


def main(argv=None):
    p = argparse.ArgumentParser(description="SER-Slice training")
    p.add_argument("--profile", choices=list(PROFILES), default="5g")
    p.add_argument("--config", default=None, help="optional base config JSON path")
    p.add_argument("--algo", choices=["serdqn", "static", "rr", "prop", "sla"], default="serdqn")
    p.add_argument("--arm", choices=["A1", "A2", "A3", "A4"], default="A3")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap steps per episode (smoke testing)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    cfg = make_config(args.profile, base_path=args.config)
    out = args.out or f"runs/{args.profile}_{args.algo}_{args.arm}_s{args.seed}"
    run(cfg, algo=args.algo, arm=args.arm, episodes=args.episodes, seed=args.seed,
        out_dir=out, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
