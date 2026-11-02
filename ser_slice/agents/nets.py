"""Pure-Python reference MLP with manual backpropagation and Adam.

This module is deliberately dependency-free: together with the environment it
is the executable reference for the SER-DQN math (DESIGN.md section 9). For
full-scale experiments, swap in a torch backend behind the same interface.
"""
import math
import random


class Dense:
    def __init__(self, n_in, n_out, rng: random.Random):
        scale = math.sqrt(2.0 / n_in)  # He init (ReLU networks)
        self.W = [[rng.gauss(0.0, scale) for _ in range(n_in)] for _ in range(n_out)]
        self.b = [0.0] * n_out
        self.dW = [[0.0] * n_in for _ in range(n_out)]
        self.db = [0.0] * n_out
        self._x = None

    def forward(self, x):
        self._x = x
        out = []
        for j in range(len(self.b)):
            s = self.b[j]
            Wj = self.W[j]
            for k, xk in enumerate(x):
                s += Wj[k] * xk
            out.append(s)
        return out

    def backward(self, dz):
        """Accumulate dW/db given dL/d(output); return dL/d(input)."""
        x = self._x
        dx = [0.0] * len(x)
        for j, dzj in enumerate(dz):
            if dzj == 0.0:
                continue
            self.db[j] += dzj
            Wj, dWj = self.W[j], self.dW[j]
            for k, xk in enumerate(x):
                dWj[k] += dzj * xk
                dx[k] += Wj[k] * dzj
        return dx

    def zero_grad(self):
        for row in self.dW:
            for k in range(len(row)):
                row[k] = 0.0
        for k in range(len(self.db)):
            self.db[k] = 0.0


class MLP:
    """MLP with ReLU hidden layers and a linear output head."""

    def __init__(self, sizes, rng: random.Random):
        self.sizes = list(sizes)
        self.layers = [Dense(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]
        self._acts = None

    def forward(self, x):
        acts = [x]
        h = x
        last = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            z = layer.forward(h)
            if idx < last:
                h = [v if v > 0.0 else 0.0 for v in z]
                acts.append(h)
            else:
                h = z
        self._acts = acts
        return h

    def backward(self, dout):
        """Accumulate parameter grads for dL/d(forward output) = `dout`.

        Requires a matching forward() call (activation caches are reused).
        """
        grad = dout
        last = len(self.layers) - 1
        for idx in range(last, -1, -1):
            layer = self.layers[idx]
            if idx < last:
                # ReLU derivative through the cached post-activation values
                grad = [g if a > 0.0 else 0.0 for g, a in zip(grad, self._acts[idx + 1])]
            grad = layer.backward(grad)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    # -------------------------------------------------------- serialization

    def get_state(self):
        return {
            "sizes": self.sizes,
            "W": [layer.W for layer in self.layers],
            "b": [layer.b for layer in self.layers],
        }

    def set_state(self, state):
        assert state["sizes"] == self.sizes
        for layer, W, b in zip(self.layers, state["W"], state["b"]):
            for j in range(len(W)):
                layer.W[j] = list(W[j])
                layer.b[j] = b[j]

    def copy_from(self, other):
        self.set_state(other.get_state())


class Adam:
    """Adam over a list of Dense layers (element-wise, in place)."""

    def __init__(self, layers, lr=5e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.layers = layers
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.mW = [[[0.0] * len(row) for row in L.W] for L in layers]
        self.vW = [[[0.0] * len(row) for row in L.W] for L in layers]
        self.mb = [[0.0] * len(L.b) for L in layers]
        self.vb = [[0.0] * len(L.b) for L in layers]

    def step(self):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for li, L in enumerate(self.layers):
            for j in range(len(L.W)):
                for k in range(len(L.W[j])):
                    g = L.dW[j][k]
                    m = self.b1 * self.mW[li][j][k] + (1.0 - self.b1) * g
                    v = self.b2 * self.vW[li][j][k] + (1.0 - self.b2) * g * g
                    self.mW[li][j][k], self.vW[li][j][k] = m, v
                    L.W[j][k] -= self.lr * (m / c1) / (math.sqrt(v / c2) + self.eps)
                g = L.db[j]
                m = self.b1 * self.mb[li][j] + (1.0 - self.b1) * g
                v = self.b2 * self.vb[li][j] + (1.0 - self.b2) * g * g
                self.mb[li][j], self.vb[li][j] = m, v
                L.b[j] -= self.lr * (m / c1) / (math.sqrt(v / c2) + self.eps)
