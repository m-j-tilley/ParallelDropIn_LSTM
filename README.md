# ParallelDropIn_LSTM

**A fast parallel warm-start for `torch.nn.LSTM`.**

Train an LSTM in parallel with a (dual-sign) Heinsen scan, then transfer the weights to
the default PyTorch LSTM implementation and finish on cuDNN.

The parallel phase uses input-only gates, so the cell update `c_t = f_t·c_{t-1} + i_t·tanh(g_t)`
is a first-order linear recurrence a scan evaluates in O(log T). That is exactly the
`W_hh = 0` case of a standard `torch.nn.LSTM`, so the trained input weights drop in with
no conversion.

## Install

```bash
pip install -e .
```

## Usage

```python
import torch
from paralleldropin_lstm import ParallelDropInLSTM

m = ParallelDropInLSTM(input_size=128, hidden_size=256)
x = torch.randn(64, 8, 128)          # [T, B, D]
out, (h_n, c_n) = m(x)               # parallel forward, O(log T) depth

lstm = m.export_to_nn_lstm()         # a stock torch.nn.LSTM with W_hh = 0
# ... keep training `lstm` with ordinary BPTT (cuDNN); W_hh grows from zero.
```

`export_to_nn_lstm()` is exact: the returned LSTM reproduces the parallel forward to
numerical precision (see `tests/test_parity.py`).

## Related work

Parallel-training an LSTM by dropping the gates' dependence on the previous hidden state
turns the update into an associative first-order recurrence a parallel prefix scan
evaluates in O(log T):

- **Were RNNs All We Needed?**, Feng et al., 2024
  ([arXiv:2410.01201](https://arxiv.org/abs/2410.01201)). *minLSTM* / *minGRU*: input-only
  gates make the recurrence a parallel scan.
- **xLSTM: Extended Long Short-Term Memory**, Beck et al., 2024
  ([arXiv:2405.04517](https://arxiv.org/abs/2405.04517)). The *mLSTM* cell is
  parallelisable by making the gates depend only on the input.
- **Heinsen, 2023** ([arXiv:2311.06281](https://arxiv.org/abs/2311.06281)): the log-domain
  associative scan; the dual-sign scan here is its signed extension.

## License

MIT.
