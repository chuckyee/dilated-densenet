from __future__ import print_function, division

import unittest
import dataset

class TestDataset(unittest.TestCase):
    def test_rvsc(self):
        d = dataset.RVSC('./test-assets')
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

        d = dataset.RVSC('./test-assets', image_transform=double,
                         mask_transform=epsilon, image_dtype='float64')
        self.assertEqual(len(d), 3)

        for i in range(3):
            image, mask = d[i]
            self.assertTupleEqual(image.shape, (1, 216, 256))
            self.assertEqual(image.dtype, 'float64')
            self.assertEqual(mask, 'epsilon')
