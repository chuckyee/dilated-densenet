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
