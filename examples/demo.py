"""End-to-end demo: train in parallel, transfer to nn.LSTM, finish with BPTT.

Toy task: predict the running mean of the first input channel. Phase 1 trains the
input-only cell in parallel (dual-sign scan); the weights are then dropped into a
stock torch.nn.LSTM, which Phase 2 keeps training with ordinary BPTT.
"""
import torch
import torch.nn as nn

from paralleldropin_lstm import ParallelDropInLSTM

torch.manual_seed(0)
D, H, T, B = 4, 32, 30, 64


def batch():
    x = torch.randn(T, B, D)
    y = torch.cumsum(x[..., :1], dim=0) / torch.arange(1, T + 1).view(T, 1, 1)   # running mean
    return x, y


def train(step_fn, params, steps, lr=1e-2):
    opt = torch.optim.Adam(params, lr=lr)
    loss = None
    for _ in range(steps):
        x, y = batch()
        loss = ((step_fn(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


readout = nn.Linear(H, 1)

# Phase 1: parallel training (O(log T) depth, no unrolling).
par = ParallelDropInLSTM(D, H)
l1 = train(lambda x: readout(par(x)[0]), list(par.parameters()) + list(readout.parameters()), 200)
print(f"phase 1 (parallel)     final loss: {l1:.5f}")

# Transfer: parallel weights -> stock torch.nn.LSTM (W_hh = 0).
lstm = par.export_to_nn_lstm()
with torch.no_grad():
    x, _ = batch()
    print(f"transfer parity max|Δ|: {(par(x)[0] - lstm(x)[0]).abs().max().item():.2e}")

# Phase 2: continue with BPTT; the recurrent weights grow from zero.
l2 = train(lambda x: readout(lstm(x)[0]), list(lstm.parameters()) + list(readout.parameters()), 200)
print(f"phase 2 (nn.LSTM BPTT) final loss: {l2:.5f}")
