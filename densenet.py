import torch
from torch import nn
import torch.nn.functional as F

from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, in_features, growth_rate, dropout_rate=0, dilation=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_features, growth_rate,
                                          kernel_size=3, stride=1,
                                          padding=dilation,
                                          dilation=dilation, bias=False))
        self.dropout_rate = dropout_rate

    def forward(self, x):
        features = super(_DenseLayer, self).forward(x)
        if self.dropout_rate > 0:
            features = F.dropout(features, p=self.dropout_rate,
                                 training=self.training)
        return torch.cat([x, features], dim=1)

class DilatedDenseNet(nn.Module):
    def __init__(self, image_channels=1, num_init_features=12, growth_rate=12,
                 layers=8, dropout_rate=0, classes=2, dilated=True):
        super(DilatedDenseNet, self).__init__()

        self.classes = classes

        self.initial_conv = nn.Conv2d(in_channels=image_channels,
                                      out_channels=num_init_features,
                                      kernel_size=5, padding=2)

        self.features = nn.Sequential(OrderedDict([]))
        nfeatures = 1 + num_init_features
        dilation = 1
        for n in range(layers):
            name = "denselayer{:02d}".format(n)
            layer = _DenseLayer(nfeatures, growth_rate, dropout_rate, dilation)
            self.features.add_module(name, layer)
            nfeatures += growth_rate
            if dilated:
                dilation *= 2

        self.logits = nn.Conv2d(in_channels=nfeatures,
                                out_channels=self.classes,
                                kernel_size=1)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.features(torch.cat([x, out], dim=1))
        return self.logits(out)

    def num_trainable_parameters(self):
        def prod(a):
            total = 1
            for x in a:
                total *= x
            return total
        return sum(prod(p.size()) for p in self.parameters() if p.requires_grad)
