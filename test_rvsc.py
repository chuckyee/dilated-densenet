from __future__ import print_function, division

import unittest
import numpy as np
import rvsc

class TestDataset(unittest.TestCase):
    def test_rvsc(self):
        d = rvsc.RVSC('./test-assets')
        self.assertEqual(len(d), 3)

        for i in range(3):
            image, mask = d[i]
            self.assertTupleEqual(image.shape, (1, 216, 256))
            self.assertTupleEqual(mask.shape, (216, 256))
            self.assertEqual(image.dtype, 'float32')
            self.assertEqual(mask.dtype, 'int64')

    def test_transform(self):
        def epsilon(x):
            return 'epsilon'

        def double(x):
            return 2*x

        d = rvsc.RVSC('./test-assets', image_transform=double,
                         mask_transform=epsilon, image_dtype='float64')
        self.assertEqual(len(d), 3)

        for i in range(3):
            image, mask = d[i]
            self.assertTupleEqual(image.shape, (1, 216, 256))
            self.assertEqual(image.dtype, 'float64')
            self.assertEqual(mask, 'epsilon')

    def test_normalize(self):
        x = np.array([[1, -1], [-1, 1]], dtype='float32')[None,:,:]
        y = rvsc.normalize(x)
        np.testing.assert_array_equal(x, y)

        # this array: mean = -0.5, std = 2.5
        x = np.array([[2, -3], [-3, 2]], dtype='float32')[None,:,:]
        y = rvsc.normalize(x)
        np.testing.assert_array_equal(y, [[[1, -1], [-1, 1]]])
