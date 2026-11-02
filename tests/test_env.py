"""Environment contract tests: action grid, reset determinism, step API,
reward bounds, truncation, and 6G-profile deltas."""
import math

from ser_slice.config import make_config
from ser_slice.env.ran_env import RanSlicingEnv, build_action_grid


def tiny(profile="5g"):
    return make_config(profile, {
        "env": {"episode_steps": 20, "ues_per_slice": [2, 2, 2], "mmtc_devices": 10},
        "traffic": {"mmtc_devices": 10, "mmtc_p_per_step": 0.05},
    })


def test_action_grid_compositions():
    grid = build_action_grid(10, 3)
    assert len(grid) == math.comb(12, 2) == 66
    assert all(sum(units) == 10 for units in grid)
    assert grid[0] == (0, 0, 10) and grid[-1] == (10, 0, 0)


def test_env_grid_allocations_within_budget():
    for profile in ("5g", "6g"):
        env = RanSlicingEnv(make_config(profile), seed=1)
        assert env.action_space.n == 66 * env.cfg["env"]["ris_modes"]
        for idx in range(66):
            units = env.grid[idx]
            assert sum(units) == env.cfg["env"]["action_units"]
            assert sum(u * env.prbs_per_unit for u in units) <= env.total_prbs


def test_reset_determinism():
    obs1, _ = RanSlicingEnv(tiny(), seed=42).reset(seed=123)
    obs2, _ = RanSlicingEnv(tiny(), seed=42).reset(seed=123)
    assert obs1 == obs2
    assert len(obs1) == RanSlicingEnv(tiny(), seed=1).obs_size


def test_step_api_and_reward_bounds():
    cfg = tiny()
    env = RanSlicingEnv(cfg, seed=7)
    obs, info = env.reset(seed=9)
    for a in (0, 33, 65):
        obs, r, terminated, truncated, info = env.step(a)
        assert len(obs) == env.obs_size
        assert env.observation_space.contains(obs), obs
        lo, hi = cfg["reward"]["clip"]
        assert lo <= r <= hi
        assert terminated is False
        assert isinstance(truncated, bool)
        assert all(math.isfinite(v) for v in obs)
        kpi = info["kpi"]
        assert all(math.isfinite(kpi[k]) for k in
                   ("thr_e_mbps", "thr_u_mbps", "thr_m_mbps", "urllc_violation_rate"))


def test_episode_truncates_at_horizon():
    env = RanSlicingEnv(tiny(), seed=3)
    env.reset(seed=5)
    for _ in range(env.episode_steps - 1):
        _, _, terminated, truncated, _ = env.step(0)
        assert not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert truncated and not terminated


def test_6g_profile_deltas():
    cfg5, cfg6 = make_config("5g"), make_config("6g")
    assert cfg6["env"]["total_prbs"] == 273 and cfg6["env"]["ris_modes"] == 2
    assert cfg6["env"]["isac"] and cfg6["traffic"]["urllc_deadline_s"] == 0.005
    env5 = RanSlicingEnv(cfg5, seed=2)
    env6 = RanSlicingEnv(cfg6, seed=2)
    assert env6.obs_size == env5.obs_size + 3  # ISAC forecast, regime, RIS flag
    assert env6.action_space.n == 2 * env5.action_space.n


def test_ris_switch_and_mode_decode():
    cfg = make_config("6g", {"env": {"episode_steps": 10, "ues_per_slice": [2, 2, 2]}})
    env = RanSlicingEnv(cfg, seed=11)
    env.reset(seed=1)
    _, _, _, _, info = env.step(3 * 2 + 0)  # split 3, ris 0
    assert info["ris_mode"] == 0
    _, _, _, _, info = env.step(3 * 2 + 1)  # split 3, ris 1 -> switching step
    assert info["ris_mode"] == 1
    _, _, _, _, info = env.step(3 * 2 + 1)  # no switch, ris active
    assert info["ris_mode"] == 1


def test_invalid_action_raises():
    env = RanSlicingEnv(tiny(), seed=1)
    env.reset(seed=1)
    try:
        env.step(env.action_space.n)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range action must raise")


def test_baselines_act_within_space():
    from ser_slice.agents.baselines import make_baseline
    env = RanSlicingEnv(tiny(), seed=4)
    obs, info = env.reset(seed=4)
    for name in ("static", "rr", "prop", "sla"):
        pol = make_baseline(name, env.cfg)
        pol.reset()
        a = pol.act(obs, info, env)
        assert env.action_space.contains(a)
