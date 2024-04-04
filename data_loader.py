import logging

from typing import Tuple, Callable, Optional, List, Iterable, Union, Literal, Sized

import numpy as np
import torch.utils.data as data
from torchvision import transforms


from .datasets import MedicalSegmentDataset
from .transforms import Normalize, ToTensor, FixedResize


def _data_transforms_medical_segmentation() -> Tuple[Callable, Callable]:
    Medical_MEAN = (0.485, 0.456, 0.406)
    Medical_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose(
        [FixedResize(512), Normalize(mean=Medical_MEAN, std=Medical_STD), ToTensor()]
    )


    return transform, transform


def load_medical_segmentation(
    data_dir: str,
    data_test_dir: str,
    train_batch_size: int,
    test_batch_size: int
) -> Iterable[Union[data.DataLoader, int]]:
    
    transform_train, transform_test = _data_transforms_medical_segmentation()

    train_ds = MedicalSegmentDataset(
        data_dir,
        name="train",
        train=True,
        transform=transform_train
    )
    train_dataLoader = data.DataLoader(
        dataset=train_ds, batch_size=train_batch_size, shuffle=True, drop_last=True
    )

 
    test_ds = MedicalSegmentDataset(
        data_test_dir, name="val", train=False, transform=transform_test, data_idxs=None
    )
    test_dataLoader = data.DataLoader(
        dataset=test_ds, batch_size=test_batch_size, shuffle=False, drop_last=True
    )

    return train_dataLoader, test_dataLoader, train_ds.num_classes, len(train_ds)




