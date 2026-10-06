"""DeepEP v2 SDAA backend."""
from .sdaa import EPBuffer, EPHandle, EventOverlap
import torch

topk_idx_t = torch.int64
__version__ = "2.5.0+sdaa"
