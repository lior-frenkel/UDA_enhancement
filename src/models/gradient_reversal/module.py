"""Gradient Reversal nn.Module wrapper."""

import torch
from torch import nn
from .functional import revgrad


class GradientReversal(nn.Module):
    """Applies gradient reversal during backward pass.

    Args:
        alpha (float): Scaling factor for the reversed gradient.
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
