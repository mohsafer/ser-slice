"""Greedy evaluation harness -> CSV traces + summary (serdqn-style).

    python -m ser_slice.eval_harness --ckpt runs/.../ckpt.json --profile 5g --episodes 100

Writes eval.csv (per-step rows) and summary.json (per-episode means).
Train on 5G, evaluate on 6G (`--profile 6g`) for the RQ3 transfer protocol.
"""
import argparse
import csv
import json
import os

from ser_slice.agents.baselines import make_baseline
from ser_slice.agents.dqn_agent import SERDQNAgent
from ser_slice.config import make_config
from ser_slice.env.profiles import PROFILES
from ser_slice.env.ran_env import RanSlicingEnv

STEP_KEYS = (
    "thr_e_mbps", "thr_u_mbps", "thr_m_mbps", "urllc_violation_rate",
    "mmtc_delay_s", "overflow_frac", "prb_util", "u_e", "u_u", "u_m",
)


def evaluate(env, act_fn, episodes=100, seed_base=777000, csv_path=None):
    writer = None
    fh = None
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        fh = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(fh)
        writer.writerow(["episode", "step", "action", "reward"] + list(STEP_KEYS))

    ep_returns = []
    kpi_acc = {k: 0.0 for k in STEP_KEYS}
    n_total = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=seed_base + ep)
        ep_ret, n = 0.0, 0
        done = False
        while not done:
            a = act_fn(obs, info, env)
            obs, r, terminated, truncated, info = env.step(a)
            ep_ret += r
            n += 1
            kpi = info["kpi"]
            if writer is not None:
                writer.writerow([ep, n, a, f"{r:.6f}"] + [f"{kpi.get(k, 0.0):.6f}" for k in STEP_KEYS])
            for k in STEP_KEYS:
                kpi_acc[k] += kpi.get(k, 0.0)
            n_total += 1
            done = terminated or truncated
        ep_returns.append(ep_ret / max(1, n))
    if fh is not None:
        fh.close()

    summary = {
        "episodes": episodes,
        "return_mean": sum(ep_returns) / len(ep_returns),
        "return_std": (sum((r - sum(ep_returns) / len(ep_returns)) ** 2 for r in ep_returns) / len(ep_returns)) ** 0.5,
        "kpi_mean": {k: kpi_acc[k] / max(1, n_total) for k in STEP_KEYS},
    }
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="SER-Slice greedy evaluation")
    p.add_argument("--ckpt", default=None, help="agent checkpoint (ckpt.json)")
    p.add_argument("--algo", choices=["serdqn", "static", "rr", "prop", "sla"],
                   default=None, help="baseline algo (if no ckpt)")
    p.add_argument("--arm", default="A3")
    p.add_argument("--profile", choices=list(PROFILES), default="5g")
    p.add_argument("--config", default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args(argv)

    cfg = make_config(args.profile, base_path=args.config)
    env = RanSlicingEnv(cfg, seed=42)

    if args.ckpt:
        from ser_slice.utils import load_json
        state = load_json(args.ckpt)
        agent = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"],
                            seed=1, arm=state.get("arm", args.arm))
        agent.load_state(state)
        act_fn = lambda obs, info, env_: agent.act(obs, greedy=True)
        tag = f"serdqn_{agent.arm}"
    else:
        policy = make_baseline(args.algo, cfg)
        act_fn = lambda obs, info, env_: policy.act(obs, info, env_)
        tag = f"baseline_{args.algo}"

    out = args.out or f"runs/eval_{args.profile}_{tag}"
    summary = evaluate(env, act_fn, episodes=args.episodes,
                       csv_path=os.path.join(out, "eval.csv"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "profile": args.profile, **summary}, fh, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "kpi_mean"}, indent=2))
    print("kpi_mean:", {k: round(v, 4) for k, v in summary["kpi_mean"].items()})


if __name__ == "__main__":
    main()
