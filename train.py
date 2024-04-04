import os
import torch
import matplotlib.pyplot as plt
#import pytorch_lightning as pl
#import segmentation_models_pytorch as smp

from pprint import pprint
#from torch.utils.data import DataLoader
from .data_loader import load_medical_segmentation
from .model import PetModel



train_batch_size=32
test_batch_size=32

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=5,
)


train_dataloader, valid_dataloader = load_medical_segmentation("/home/train",
    "/home/test",
    train_batch_size,
    test_batch_size)


model = PetModel("FPN", "resnet34", in_channels=3, out_classes=1)

trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
    val_dataloaders=valid_dataloader,
)




valid_metrics = trainer.validate(model, dataloaders=valid_dataloader, verbose=False)
pprint(valid_metrics)


#test_metrics = trainer.test(model, dataloaders=test_dataloader, verbose=False)
#pprint(test_metrics)