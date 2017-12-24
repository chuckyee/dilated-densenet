from __future__ import print_function, division

import unittest
from math import sqrt

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable

import densenet


class TestDenseNet(unittest.TestCase):
    def compute_batch_norm_relu(self, x):
        # compute batchnorm with gamma = 1, beta = 0
        a = Tensor(x)
        # first batch norm
        epsilon = 1e-5          # default value from paper
        # biased estimator for variance (1/N norm used, not 1/(N-1))
        b = (a - a.mean()) / sqrt(a.var(unbiased=False) + epsilon)
        # then relu
        b = torch.max(b, Tensor([0]))
        return b

    def test_denselayer(self):
        # Tests forward pass through individual BN-ReLU-Conv2D layer

        layer = densenet._DenseLayer(in_features=1, growth_rate=1)

        # explicitly set batchnorm gamma = 1 and beta = 0
        layer.norm.weight.data[0] = 1
        layer.norm.bias.data[0] = 0

        # explicitly set convolutional kernel to just identity
        kernel = Tensor([
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
        x = Variable(Tensor([[image]]))
        y = layer(x)

        # construct what we expect output to be: just batch norm and relu
        # (identity conv2d does nothing)
        b = self.compute_batch_norm_relu(image)

        # first output map is just original input
        self.assertTrue(torch.equal(y[0,0].data, x.data[0,0]))
        # second output map is the batchnorm-relu-conv2d
        # self.assertTrue(torch.equal(y[0,1].data, b))
        self.assertTrue(torch.equal(y[0,1].data, b))

    def test_densenet(self):
        # Tests forward pass through small dilated densenet network
        # WARNING: doesn't test dilated convs since we use kernel = identity

        model = densenet.DilatedDenseNet(image_channels=1, num_init_features=1,
                                         growth_rate=1, layers=2, dilated=True)

        # input feature map
        image = [
            [1,  0, 1, 0],
            [0, -1, 1, 0],
            [1,  0, 2, 1],
            [1, -1, 1, 1],
        ]
        x = Variable(Tensor([[image]]))

        initial_conv = model.initial_conv
        initial_conv.weight.data[0,0,:,:] = Tensor([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        initial_conv.bias.data = Tensor([0])

        layer00 = model.features.denselayer00
        for i in range(2):
            layer00.norm.weight.data[i] = 1
            layer00.norm.bias.data[i] = 0
            kernel = Tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ])
            layer00.conv.weight.data[0,i,:,:] = kernel

        layer01 = model.features.denselayer01
        for i in range(3):
            layer01.norm.weight.data[i] = 1
            layer01.norm.bias.data[i] = 0
            kernel = Tensor([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ])
            layer01.conv.weight.data[0,i,:,:] = kernel

        logits = model.logits
        logits.weight.data[0,:,0,0] = Tensor([1, 1, 1, 1])
        logits.weight.data[1,:,0,0] = Tensor([-1, -1, -1, -1])
        logits.bias.data = Tensor([0, 0])

        # initial 5x5 convolution
        x1 = initial_conv(x)
        self.assertTrue(torch.equal(x, x1)) # initial conv is identity

        # dense connection concats input with output of initial conv
        x2 = layer00(torch.cat([x, x1], dim=1)) # concat along channel dimen.

        # second dense connection; layer00 contains concat operation
        x3 = layer01(x2)

        # logits are 1x1 conv to 2 masks
        x4 = logits(x3)

        # first dense layer: output is 3 feature maps.
        # Define RBN(x) = ReLU(BN(x)) is relu + batch norm operation
        #  - first = input
        #  - last = 2 BNR(x)
        b = self.compute_batch_norm_relu(image)
        self.assertTrue(x2.size() == torch.Size([1, 3, 4, 4]))
        self.assertTrue(torch.equal(x[0,0], x2[0,0]))
        self.assertTrue(torch.equal(x[0,0], x2[0,1]))
        self.assertTrue(torch.equal(2*b, x2[0,2].data))

        # second dense layer: output is 4 feature maps
        #  - first two = input image
        #  - third = 2 BNR(x)
        #  - fourth = 2 BNR(x) + BNR(BNR(x))
        c = self.compute_batch_norm_relu(b)
        self.assertTrue(x3.size() == torch.Size([1, 4, 4, 4]))
        self.assertTrue(torch.equal(x[0,0], x3[0,0]))
        self.assertTrue(torch.equal(x[0,0], x3[0,1]))
        self.assertTrue(torch.equal(2*b, x3[0,2].data))
        # float 32 precision loss limits comparison to 4th decimal
        np.testing.assert_almost_equal(
            np.asarray(2*b + c), np.asarray(x3[0,3].data),
            decimal=4)

        d = torch.sum(x3, dim=1)[0]
        self.assertTrue(torch.equal(d, x4[0,0]))
        self.assertTrue(torch.equal(-d, x4[0,1]))

        # test full model end-to-end
        y = model(x)
        self.assertTrue(torch.equal(d, y[0,0]))
        self.assertTrue(torch.equal(-d, y[0,1]))
