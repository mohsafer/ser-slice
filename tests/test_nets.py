"""Reference-MLP correctness: analytic gradients vs numerical gradients,
and an Adam convergence sanity check. The whole SER-DQN rests on these."""
import math
import random

from ser_slice.agents.nets import MLP, Adam


def mse_loss_and_grad(net, x, target):
    out = net.forward(x)
    loss = sum((o - t) ** 2 for o, t in zip(out, target)) / len(out)
    dout = [(2.0 / len(out)) * (o - t) for o, t in zip(out, target)]
    net.zero_grad()
    net.backward(dout)
    return loss


def test_gradient_check():
    rng = random.Random(0)
    net = MLP([3, 4, 2], rng)
    x = [0.4, -1.2, 0.7]
    target = [0.5, -0.3]
    eps = 1e-5
    worst = 0.0
    for li, layer in enumerate(net.layers):
        for j in range(len(layer.W)):
            for k in range(len(layer.W[j])):
                orig = layer.W[j][k]
                layer.W[j][k] = orig + eps
                lp = mse_loss_and_grad(net, x, target)
                layer.W[j][k] = orig - eps
                lm = mse_loss_and_grad(net, x, target)
                layer.W[j][k] = orig
                numeric = (lp - lm) / (2 * eps)
                mse_loss_and_grad(net, x, target)  # restore analytic grads
                analytic = layer.dW[j][k]
                denom = max(1.0, abs(numeric), abs(analytic))
                worst = max(worst, abs(numeric - analytic) / denom)
        for j in range(len(layer.b)):
            orig = layer.b[j]
            layer.b[j] = orig + eps
            lp = mse_loss_and_grad(net, x, target)
            layer.b[j] = orig - eps
            lm = mse_loss_and_grad(net, x, target)
            layer.b[j] = orig
            numeric = (lp - lm) / (2 * eps)
            mse_loss_and_grad(net, x, target)
            analytic = layer.db[j]
            denom = max(1.0, abs(numeric), abs(analytic))
            worst = max(worst, abs(numeric - analytic) / denom)
    assert worst < 1e-5, f"gradient check failed, worst rel err {worst}"


def test_adam_converges_on_toy_regression():
    rng = random.Random(1)
    net = MLP([1, 8, 1], rng)
    opt = Adam(net.layers, lr=0.02)
    data = [(x / 5.0, 0.7 * (x / 5.0) + 0.2) for x in range(-5, 6)]
    first_loss = None
    for _ in range(300):
        epoch_loss = 0.0
        net.zero_grad()
        for x, y in data:
            out = net.forward([x])
            err = out[0] - y
            epoch_loss += 0.5 * err * err
            net.backward([err])
        first_loss = first_loss if first_loss is not None else epoch_loss
        opt.step()
    final = 0.0
    for x, y in data:
        out = net.forward([x])
        final += 0.5 * (out[0] - y) ** 2
    assert final < 1e-2, f"loss did not converge: first={first_loss}, final={final}"


def test_state_roundtrip():
    rng = random.Random(2)
    net = MLP([4, 5, 3], rng)
    net2 = MLP([4, 5, 3], random.Random(99))
    net2.copy_from(net)
    x = [0.1, 0.2, -0.3, 0.5]
    assert net.forward(x) == net2.forward(x)
