"""ParallelDropIn_LSTM.

Train an LSTM in parallel with a (dual-sign) Heinsen scan, then transfer the weights
to the default PyTorch LSTM implementation.
"""
from .model import ParallelDropInLSTM
from .scan import dual_sign_scan

__version__ = "0.1.0"
__all__ = ["ParallelDropInLSTM", "dual_sign_scan", "__version__"]
