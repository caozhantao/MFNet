# Code reference: https://github.com/deepchecks/deepchecks/blob/daedbaba3ba0e020e96de2ac0eb6a6f24d5359c5/docs/source/user-guide/vision/tutorials/plot_custom_task_tutorial.py

import contextlib
import os
import typing as t
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.utils import draw_segmentation_masks


class MedicalSegmentDataset(VisionDataset):
    """An instance of PyTorch VisionData the represents the medical-segments dataset.
    Parameters
    ----------
    root : str
        Path to the root directory of the dataset.
    name : str
        Name of the dataset.
    train : bool
        if `True` train dataset, otherwise test dataset
    transforms : Callable, optional
        A function/transform that takes in an PIL image and returns a transformed version.
        E.g, transforms.RandomCrop
    """

    TRAIN_FRACTION = 1

    def __init__(
        self,
        root: str,
        name: str = "train",
        train: bool = True,
        transform: t.Optional[t.Callable] = None,
    ) -> None:
        super().__init__(root, transforms=transform)

        self.train = train
        self.root = Path(root).absolute()
        self.images_dir = Path(root) / name / "images" 
        self.labels_dir = Path(root) / name / "masks"
        self.num_classes = 2
        
        self.targets = None


        images: t.List[Path] = sorted(self.images_dir.glob("./*.jpg"))
        labels: t.List[t.Optional[Path]] = []

        for image in images:
            label = self.labels_dir / f"{image.stem}.jpg"
            labels.append(label if label.exists() else None)

        assert len(images) != 0, "Did not find folder with images or it was empty"
        assert not all(
            l is None for l in labels
        ), "Did not find folder with labels or it was empty"

        train_len = int(self.TRAIN_FRACTION * len(images))

        #self.images = images[0:train_len]
        #self.labels = labels[0:train_len]
        self.images = images
        self.labels = labels

        self.__generate_targets()

    def __generate_targets(self):
        """
        Used to generate targets which in turn is used to partition data in an non-IID setting.
        """
        targets = list()
        for i in range(len(self.images)):
            categories = [0, 1]
            if isinstance(categories, np.ndarray):
                categories = np.asarray(list(categories))
            else:
                categories = np.asarray(categories).astype(np.uint8)
            targets.append(categories)
        self.targets = np.asarray(targets)

    def __getitem__(self, idx: int) -> t.Tuple[Image.Image, np.ndarray]:
        """Get the image and label at the given index."""
        image = Image.open(str(self.images[idx]))
        # to RGB
        image = image.convert("RGB")
        #np.set_printoptions(threshold=np.inf)
        #print("*******openimage****************")
        #print(np.array(image))

        label_file = self.labels[idx]

        masks = []
        classes = []
        #np.set_printoptions(threshold=np.inf)

        if label_file is not None:
            mask = Image.open(str(label_file))
            
            mask = mask.convert("L")
           
            mask = np.array(mask)
            mask[mask < 128] = 0
            mask[mask >= 128] = 1
            
            masks = mask
     
        if self.transforms is not None:
            transformed = self.transforms(
                {
                    "image": image,
                    "label": masks,
                    "class_id": classes,
                    "class_num": self.num_classes,
                }
            )
            image = transformed["image"]
            masks = transformed["label"]

       

        return {"image": image, "label": masks, "class": classes}

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """
        Returns:
            The clasess present in the Medical dataset.
        """
        return ('__background__', 'tumour')


if __name__ == "__main__":
    root_dir = "/home/beiyu/fedcv_data/medical"
    train_data = MedicalSegmentDataset(root=root_dir, train=True)
    print(len(train_data))
    print(train_data.__getitem__(0))
