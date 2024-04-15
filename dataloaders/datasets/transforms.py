import logging
import random
from typing import Iterable, Dict

import numpy as np
import torch

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    mean: Iterable[float]
    std: Iterable[float]

    def __init__(
        self,
        mean: Iterable[float] = (0.0, 0.0, 0.0),
        std: Iterable[float] = (1.0, 1.0, 1.0),
    ):
        """
        Normalizes using the mean and standard deviation
        Args:
            mean: The mean value of the distribution
            std: The standard value of the distribution
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, np.ndarray]:
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        # logging.info(f"Image shape: {img.shape}")

        img /= 255.0
        img -= self.mean
        img /= self.std

        sample = {'image': img, 'label': mask}
        return sample


class ToTensor(object):
    """
    Transforms the numpy array to torch tensor
    """

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, torch.FloatTensor]:
        img = sample["image"]
        mask = sample["label"]
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        sample = {'image': img, 'label': mask}
        return sample


class FixedResize(object):
    def __init__(self, size: int):
        """
        Resizes the image to the specified size.

        Args:
            size: The size to resize the image.
        """
        self.size = (size, size)

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, Image.Image]:
        img = sample["image"]
        mask = sample["label"]

        # assert img.size == mask.size

        img_size = list(img.size)[::-1]
        img = img.resize(self.size, Image.BILINEAR)

        # mask: List[List[bool, bool]], class_id: List[int], class_num: int
        # mask to multi-class mask
        mask = np.array(mask)
        #mask_ = np.zeros(img_size, dtype=np.float32)

        #for idx, class_idx in enumerate(class_id):
        #    mask_[mask[idx] == True] = class_idx

        # to Image
        # print(mask_.shape)
        #mask = Image.fromarray(mask_.astype(np.uint8))
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize(self.size, Image.NEAREST)

        sample = {'image': img, 'label': mask}
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1 * self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}