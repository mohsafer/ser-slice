"""Agent tests: all four SER arms learn finite losses, act within the space,
checkpoint round-trips, and the training pipeline runs end-to-end."""
import math
import os

from ser_slice.agents.dqn_agent import ARMS, SERDQNAgent
from ser_slice.config import make_config
from ser_slice.env.ran_env import RanSlicingEnv
from ser_slice.train import run

AGENT_CFG = {
    "agent": {
        "hidden": [16], "lr": 0.001, "gamma": 0.99,
        "eps_start": 1.0, "eps_end": 0.05, "eps_decay_steps": 100,
        "replay_capacity": 200, "batch_size": 4, "warmup_steps": 10,
        "target_sync_every": 20, "huber_delta": 1.0,
        "soft_alpha": 0.5, "diffusion_sigma": 0.15,
    }
}
ENV_CFG = {
    "env": {"episode_steps": 30, "ues_per_slice": [2, 2, 2]},
    "traffic": {"mmtc_devices": 10, "mmtc_p_per_step": 0.05},
}


def smoke_env_cfg():
    return make_config("5g", {**ENV_CFG, **AGENT_CFG})


def collect_transitions(env, agent, n=40):
    obs, info = env.reset(seed=13)
    for _ in range(n):
        a = agent.act(obs)
        obs2, r, term, trunc, info = env.step(a)
        agent.store(obs, a, r, obs2, term or trunc)
        obs = obs2
        if term or trunc:
            obs, info = env.reset()


def test_all_arms_learn_finite():
    for arm in ARMS:
        cfg = smoke_env_cfg()
        env = RanSlicingEnv(cfg, seed=5)
        agent = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=6, arm=arm)
        collect_transitions(env, agent)
        for _ in range(3):
            stats = agent.learn()
            assert stats is not None, arm
            assert math.isfinite(stats["loss"]) and stats["loss"] >= 0.0, arm
            assert math.isfinite(stats["mean_abs_advantage"]), arm
        assert env.action_space.contains(agent.act(env.reset(seed=1)[0], greedy=True))


def test_soft_vs_hard_arms_differ_in_target_math():
    """A2 (soft) and A1 (hard) must produce different targets for the same batch."""
    cfg = smoke_env_cfg()
    outs = {}
    for arm in ("A1", "A2"):
        env = RanSlicingEnv(cfg, seed=5)
        agent = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=6, arm=arm)
        collect_transitions(env, agent)
        outs[arm] = agent.learn()["loss"]
    assert outs["A1"] != outs["A2"]  # same seed/data, different target rule


def test_checkpoint_roundtrip_preserves_greedy_policy():
    cfg = smoke_env_cfg()
    env = RanSlicingEnv(cfg, seed=5)
    agent = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=6, arm="A3")
    collect_transitions(env, agent, n=60)
    for _ in range(5):
        agent.learn()
    obs, _ = env.reset(seed=77)
    before = [agent.act(obs, greedy=True) for _ in range(1)]
    state = agent.save_state()
    agent2 = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=99, arm="A3")
    agent2.load_state(state)
    assert [agent2.act(obs, greedy=True) for _ in range(1)] == before
    try:
        agent_bad = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=1, arm="A1")
        agent_bad.load_state(state)
    except AssertionError:
        pass
    else:
        raise AssertionError("cross-arm load must be rejected")


def test_train_pipeline_end_to_end(tmp_path):
    cfg = smoke_env_cfg()
    out = str(tmp_path / "run")
    for algo, arm in (("serdqn", "A3"), ("prop", None)):
        rec = run(cfg, algo=algo, arm=arm, episodes=2, seed=3, out_dir=out,
                  max_steps=15, verbose=False)
        assert rec is not None and math.isfinite(rec["return"])
        assert os.path.exists(os.path.join(out, "train.jsonl"))
        assert os.path.exists(os.path.join(out, "config.json"))
    assert os.path.exists(os.path.join(out, "ckpt.json"))


def test_eval_harness_smoke(tmp_path):
    from ser_slice.eval_harness import evaluate
    cfg = smoke_env_cfg()
    env = RanSlicingEnv(cfg, seed=5)
    agent = SERDQNAgent(env.obs_size, env.action_space.n, cfg["agent"], seed=6, arm="A3")
    summary = evaluate(env, lambda obs, info, env_: agent.act(obs, greedy=True),
                       episodes=3, csv_path=str(tmp_path / "eval.csv"))
    assert summary["episodes"] == 3
    assert math.isfinite(summary["return_mean"])
    assert os.path.getsize(tmp_path / "eval.csv") > 0
