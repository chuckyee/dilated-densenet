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

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class RVSC(Dataset):
    r"""Right ventricle segmentation challenge dataset.
    """
    def __init__(self, root, image_transform=None, mask_transform=None,
                 image_dtype='float32', mask_dtype='int64'):
        self.images_root = os.path.join(root, 'images')
        self.masks_root  = os.path.join(root, 'masks')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.masks_root) if f.endswith('.png')]
        self.filenames.sort()

        self.image_transform = image_transform
        self.mask_transform  = mask_transform

        self.image_dtype = image_dtype
        self.mask_dtype  = mask_dtype

    def __getitem__(self, index):
        filename = self.filenames[index] + '.png'

        with open(os.path.join(self.images_root, filename), 'rb') as f:
            image = np.asarray(Image.open(f), dtype=self.image_dtype)[None,:,:]
        with open(os.path.join(self.masks_root, filename), 'rb') as f:
            mask = np.asarray(Image.open(f), dtype=self.mask_dtype)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.filenames)
