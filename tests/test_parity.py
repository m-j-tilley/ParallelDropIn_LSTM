"""The core guarantee: the parallel dual-sign scan reproduces a stock torch.nn.LSTM
(with W_hh = 0) to numerical precision, so the weights transfer with no conversion.
"""
import torch

from paralleldropin_lstm import ParallelDropInLSTM


def _max_abs_diff(model, x):
    out, (h_n, c_n) = model(x)
    lstm = model.export_to_nn_lstm()
    ref, (rh, rc) = lstm(x)
    return (out - ref).abs().max().item(), (c_n - rc).abs().max().item()


def test_parity_float64():
    torch.manual_seed(0)
    m = ParallelDropInLSTM(6, 5).double()
    x = torch.randn(50, 4, 6, dtype=torch.float64)     # [T, B, D]; T=50 underflows forget products
    d_out, d_c = _max_abs_diff(m, x)
    assert d_out < 1e-10 and d_c < 1e-10, (d_out, d_c)


def test_parity_float32():
    torch.manual_seed(1)
    m = ParallelDropInLSTM(8, 7)
    x = torch.randn(64, 3, 8)
    d_out, d_c = _max_abs_diff(m, x)
    assert d_out < 1e-4 and d_c < 1e-4, (d_out, d_c)


def test_parity_batch_first():
    torch.manual_seed(2)
    m = ParallelDropInLSTM(5, 6, batch_first=True).double()
    x = torch.randn(3, 40, 5, dtype=torch.float64)     # [B, T, D]
    d_out, d_c = _max_abs_diff(m, x)
    assert d_out < 1e-10, d_out


if __name__ == "__main__":
    for dt, name in [(torch.float64, "float64"), (torch.float32, "float32")]:
        torch.manual_seed(0)
        m = ParallelDropInLSTM(6, 5).to(dt)
        x = torch.randn(50, 4, 6, dtype=dt)
        print(f"{name}: max|Δout|, max|Δc_n| =", _max_abs_diff(m, x))
    print("OK")
