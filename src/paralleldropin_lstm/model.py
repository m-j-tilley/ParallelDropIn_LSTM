"""ParallelDropInLSTM: an input-only LSTM cell trained in parallel via the
dual-sign scan, whose weights drop straight into ``torch.nn.LSTM``.

In the parallel phase the gates depend only on the input, so the cell-state update
``c_t = f_t * c_{t-1} + i_t * tanh(g_t)`` is a first-order linear recurrence that the
dual-sign scan evaluates in O(log T). That is exactly the ``W_hh = 0`` special case of
a standard ``torch.nn.LSTM`` (same gate equations, same ``tanh`` cell gate), so the
trained input weights transfer with no conversion: ``export_to_nn_lstm()`` returns a
stock LSTM you can then finish training with cuDNN BPTT (the recurrent weights grow
from zero).

Gate order matches ``torch.nn.LSTM``: ``[i, f, g, o]`` (input, forget, cell, output).
"""
import math

import torch
import torch.nn as nn

from .scan import dual_sign_scan


class ParallelDropInLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        # Only the input projection is learned in the parallel phase; the recurrent
        # weights are implicitly zero and appear when you export to nn.LSTM.
        self.weight_ih_l0 = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.bias_ih_l0 = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.LSTM's default init so the two are directly comparable.
        stdv = 1.0 / math.sqrt(self.hidden_size)
        nn.init.uniform_(self.weight_ih_l0, -stdv, stdv)
        nn.init.uniform_(self.bias_ih_l0, -stdv, stdv)

    def forward(self, x: torch.Tensor, hx=None):
        if hx is not None:
            raise NotImplementedError("v1 assumes zero initial state (hx=None)")
        if self.batch_first:
            x = x.transpose(0, 1)                       # [B, T, D] -> [T, B, D]
        H = self.hidden_size

        gates = x @ self.weight_ih_l0.t() + self.bias_ih_l0        # [T, B, 4H]
        i, f, g, o = gates.split(H, dim=-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)

        c = dual_sign_scan(f, i * g)                    # [T, B, H], c_0 = 0
        h = o * torch.tanh(c)

        out = h.transpose(0, 1) if self.batch_first else h
        return out, (h[-1:].contiguous(), c[-1:].contiguous())     # h_n, c_n : [1, B, H]

    @torch.no_grad()
    def export_to_nn_lstm(self) -> nn.LSTM:
        """Return a ``torch.nn.LSTM`` carrying these input weights with ``W_hh = 0``.

        Running the returned LSTM recurrently reproduces this module's forward pass
        exactly (see tests/test_parity.py). Continue training it with BPTT to let the
        recurrent weights grow from zero.
        """
        lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=self.batch_first)
        lstm.to(device=self.weight_ih_l0.device, dtype=self.weight_ih_l0.dtype)
        lstm.weight_ih_l0.copy_(self.weight_ih_l0)
        lstm.bias_ih_l0.copy_(self.bias_ih_l0)
        lstm.weight_hh_l0.zero_()
        lstm.bias_hh_l0.zero_()
        return lstm
