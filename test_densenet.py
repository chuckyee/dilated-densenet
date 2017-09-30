from __future__ import print_function, division

import unittest
from math import sqrt
import torch
from torch.autograd import Variable
import densenet

class TestDenseNet(unittest.TestCase):
    def test_denselayer(self):
        layer = densenet._DenseLayer(in_features=1, growth_rate=1)

        # explicitly set batchnorm gamma = 1 and beta = 0
        layer.norm.weight.data[0] = 1
        layer.norm.bias.data[0] = 0

        # explicitly set convolutional kernel to just identity
        kernel = torch.Tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        layer.conv.weight.data[0,0,:,:] = kernel

        # input feature map
        image = [
            [1,  0, 1, 0],
            [0, -1, 1, 0],
            [1,  0, 2, 1],
            [1, -1, 1, 1],
        ]
        x = Variable(torch.Tensor([[image]]))
        y = layer(x)

        # construct what we expect output to be
        a = torch.Tensor(image)
        # first batch norm
        epsilon = 1e-5
        b = (a - a.mean()) / sqrt(a.var()*15/16 + epsilon)
        # then relu
        b = torch.max(b, torch.Tensor([0]))
        # (identity conv2d does nothing)

        # first output map is just original input
        self.assertTrue(torch.equal(y[0,0].data, x.data[0,0]))
        # second output map is the batchnorm-relu-conv2d
        self.assertTrue(torch.equal(y[0,1].data, b))

    def test_densenet(self):
        model = densenet.DilatedDenseNet(image_channels=1, num_init_features=1,
                                         growth_rate=1, layers=3, dilated=True)

        # input feature map
        image = [
            [1,  0, 1, 0],
            [0, -1, 1, 0],
            [1,  0, 2, 1],
            [1, -1, 1, 1],
        ]
        x = Variable(torch.Tensor([[image]]))

        layer = model.features.denselayer02
        layer.norm.weight.data[0] = 1
        layer.norm.bias.data[0] = 0
        kernel = torch.Tensor([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ])
        layer.conv.weight.data[0,0,:,:] = kernel
        y = model(x)

        print(model)
        print(y)
