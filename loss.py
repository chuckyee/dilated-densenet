import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    r"""This criterion is the CrossEntropyLoss for 2d.

    Examples::
        >>> weight = torch.ones(2)
        >>> loss = CrossEntropyLoss2d(weight)
        >>> input = (N, C, H, W)
        >>> target = (N, H, W)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, weight=None):
        super().__init__()
        self.loss = nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs), targets)


class SoftDice2d(nn.Module):
    r"""This criterion is the `soft' dice loss for 2d, also known as the F1
    score.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.loss = None        # TODO

    def forward(self, outputs, targets):
        intersection = outputs * targets # TODO: one-hot encode
        area1 = outputs.sum(dim=2).sum(dim=2)
        area2 = targets.sum(dim=2).sum(dim=2)   # Won't work
        return 2*intersection / (area1 + area2) # TODO: regularize
