"""SER-DQN: stochastic entropy-regularized Double DQN for RAN slicing.

The target rule is the serdqn core (DESIGN.md section 5.2):

    a*     = argmax_a Q_online(s', a)                     (DDQN selection)
    V(s')  = alpha * logsumexp_a(Q_target(s', a)/alpha)   (soft)  |  max_a     (hard)
    y      = r + gamma*(1-done) * [ V(s') + (Q_target(s',a*) - V(s')) * (1 + g*|dW|) ]
    dW ~ N(0,1)                                           (Brownian diffusion)

Arms (semantics identical to the SER-DQN paper's ablation):

    A1  vanilla DDQN : hard value,   no diffusion
    A2  soft-only    : soft value,   no diffusion
    A3  full SER     : soft value,   diffusion
    A4  gain-only    : hard value,   diffusion
"""
import math
import random

from ser_slice.agents.nets import MLP, Adam
from ser_slice.utils import LinearSchedule

ARMS = {
    "A1": {"soft": False, "sigma": 0.0},
    "A2": {"soft": True, "sigma": 0.0},
    "A3": {"soft": True, "sigma": None},  # None -> take from config
    "A4": {"soft": False, "sigma": None},
}


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.pos = 0

    def __len__(self):
        return len(self.data)

    def append(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, rng, batch):
        return rng.sample(self.data, batch)


def _soft_value(q_list, alpha):
    """alpha * logsumexp(q/alpha), numerically stabilized."""
    scaled = [q / alpha for q in q_list]
    m = max(scaled)
    lse = m + math.log(sum(math.exp(s - m) for s in scaled))
    return alpha * lse


class SERDQNAgent:
    def __init__(self, n_obs, n_act, agent_cfg, seed=0, arm="A3"):
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
        self.rng = random.Random(seed)
        self.n_act = n_act
        cfg = agent_cfg
        self.gamma = cfg["gamma"]
        self.batch = cfg["batch_size"]
        self.warmup = cfg["warmup_steps"]
        self.sync_every = cfg["target_sync_every"]
        self.huber_delta = cfg["huber_delta"]
        self.soft_alpha = cfg["soft_alpha"]
        self.arm = arm

        arm_cfg = ARMS[arm]
        self.soft = arm_cfg["soft"]
        if arm_cfg["sigma"] is None:  # A3/A4 take sigma from config
            self.sigma = cfg["diffusion_sigma"]
        else:                          # A1/A2 pin it off
            self.sigma = arm_cfg["sigma"]

        self.online = MLP([n_obs] + list(cfg["hidden"]) + [n_act], self.rng)
        self.target = MLP([n_obs] + list(cfg["hidden"]) + [n_act], self.rng)
        self.target.copy_from(self.online)
        self.opt = Adam(self.online.layers, lr=cfg["lr"])
        self.buffer = ReplayBuffer(cfg["replay_capacity"])
        self.eps = LinearSchedule(cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])
        self.learn_steps = 0

    # ------------------------------------------------------------ behavior

    def act(self, obs, greedy=False):
        if not greedy and self.rng.random() < self.eps.value():
            self.eps.step()
            return self.rng.randrange(self.n_act)
        self.eps.step()
        q = self.online.forward(obs)
        return max(range(self.n_act), key=lambda a: q[a])

    def store(self, s, a, r, s2, done):
        self.buffer.append((list(s), int(a), float(r), list(s2), bool(done)))

    # ------------------------------------------------------------- learning

    def learn(self):
        if len(self.buffer) < self.warmup:
            return None
        if self.learn_steps % self.sync_every == 0:
            self.target.copy_from(self.online)

        batch = self.buffer.sample(self.rng, self.batch)
        total_loss, total_adv = 0.0, 0.0
        for s, a, r, s2, done in batch:
            q_next = self.target.forward(s2)
            a_star = max(range(self.n_act), key=lambda x: self.online.forward(s2)[x])
            q_star = q_next[a_star]
            if self.soft:
                v = _soft_value(q_next, self.soft_alpha)
            else:
                v = max(q_next)
            if self.sigma > 0.0:
                dw = abs(self.rng.gauss(0.0, 1.0))
                v_target = v + (q_star - v) * (1.0 + self.sigma * dw)
            else:
                v_target = v
            y = r + self.gamma * (0.0 if done else 1.0) * v_target
            total_adv += abs(q_star - v)

            q_all = self.online.forward(s)
            err = q_all[a] - y
            ad = abs(err)
            if ad <= self.huber_delta:
                loss = 0.5 * err * err
                derr = err
            else:
                loss = self.huber_delta * (ad - 0.5 * self.huber_delta)
                derr = self.huber_delta if err > 0 else -self.huber_delta
            total_loss += loss

            dout = [0.0] * self.n_act
            dout[a] = derr
            self.online.backward(dout)

        self.opt.step()
        self.online.zero_grad()
        self.learn_steps += 1
        return {"loss": total_loss / len(batch), "mean_abs_advantage": total_adv / len(batch)}

    # -------------------------------------------------------- serialization

    def save_state(self):
        return {
            "arm": self.arm,
            "learn_steps": self.learn_steps,
            "eps_t": self.eps.t,
            "online": self.online.get_state(),
            "target": self.target.get_state(),
        }

    def load_state(self, state):
        assert state["arm"] == self.arm, "cannot load a checkpoint from another arm"
        self.online.set_state(state["online"])
        self.target.set_state(state["target"])
        self.learn_steps = state["learn_steps"]
        self.eps.t = state["eps_t"]
