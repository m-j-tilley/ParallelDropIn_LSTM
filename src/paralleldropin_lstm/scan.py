"""Dual-sign (complex-log) parallel associative scan.

Solves the first-order linear recurrence

    c_t = a_t * c_{t-1} + b_t,     c_0 = 0,   t = 1 .. T

with

  * a_t > 0     (the forget gate; positive, so log a_t is real), and
  * b_t signed  (= i_t * tanh(g_t); its sign is carried through a *complex* log,
                 log(-x) = log|x| + i*pi -- this is the "dual sign" part).

Everything is done in the log domain so the vanishing product of forget gates does
not underflow, and the cumulative sum is a Hillis-Steele prefix scan: O(log T)
sequential depth, pure PyTorch, runs on any device. This is the signed extension of
Heinsen's log-domain scan (arXiv:2311.06281).
"""
import torch

_COMPLEX_OF = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def _logaddexp_complex(u, v):
    m = torch.maximum(u.real, v.real).to(u.dtype)
    return m + torch.log(torch.exp(u - m) + torch.exp(v - m))


def _complex_logcumsumexp(z):
    """Inclusive log-cumsum-exp over dim 0 for complex ``z`` (Hillis-Steele, O(log T))."""
    T = z.shape[0]
    d = 1
    while d < T:
        z = torch.cat([z[:d], _logaddexp_complex(z[d:], z[:-d])], dim=0)
        d *= 2
    return z


def dual_sign_scan(a, b):
    """Run the scan for ``c_t = a_t * c_{t-1} + b_t`` with ``c_0 = 0``.

    Parameters
    ----------
    a, b : torch.Tensor
        Real tensors of shape ``[T, ...]``. ``a`` must be positive; ``b`` may be
        signed. Any device; float32 or float64.

    Returns
    -------
    torch.Tensor
        ``c`` of shape ``[T, ...]`` and the same real dtype as ``b``.
    """
    cdtype = _COMPLEX_OF.get(b.dtype, torch.complex64)
    a_star = torch.cumsum(torch.log(a), dim=0)                 # log prod_{k<=t} a_k  (real)
    z = torch.log(b.to(cdtype)) - a_star.to(cdtype)            # signed via complex log
    log_c = a_star.to(cdtype) + _complex_logcumsumexp(z)
    return torch.exp(log_c).real
