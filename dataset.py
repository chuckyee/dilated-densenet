import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def normalize(x):
    r"""Zero the mean and set standard deviation to one.

    Arguments:
        x (NumPy array): array of shape (height, width)
    """
    return (x - np.mean(x.flat)) / np.std(x.flat)

class SegmentationDataset(Dataset):
    r"""Segmentation dataset. This object expects a directory with the following
    structure:
    root/
      images/
        000.jpeg
        001.jpeg
        ...
      masks/
        000.png
        001.png
        ...
    The images can be either png or jpeg format. The masks should be png
    images with integer pixel values.
    """

    extensions = ('.png', '.jpg', '.jpeg')

    def __init__(self, root, image_transform=None, mask_transform=None,
                 image_dtype='float32', mask_dtype='int32'):
        self.images_root = os.path.join(root, 'images')
        self.masks_root  = os.path.join(root, 'masks')

        self.filenames = [f for f in os.listdir(self.masks_root) if
                          f.endswith(SegmentationDataset.extensions)]
        self.filenames.sort()

        self.image_transform = image_transform
        self.mask_transform  = mask_transform

        self.image_dtype = image_dtype
        self.mask_dtype  = mask_dtype

    def __getitem__(self, index):
        filename = self.filenames[index]
        basename = os.path.basename(os.path.splitext(filename)[0])
        maskname = basename + '.png'

        filepath = os.path.join(self.images_root, filename)
        maskpath = os.path.join(self.masks_root, maskname)
        
        image = np.asarray(Image.open(filepath), dtype=self.image_dtype)
        mask = np.asarray(Image.open(maskpath), dtype=self.mask_dtype)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.filenames)
