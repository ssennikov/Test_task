import torch
import albumentations as A
import os
from torch.utils.data import DataLoader
from torch import nn
from albumentations.pytorch import ToTensorV2
from model import DeepLabv3
from catalyst import dl
from catalyst.dl import SupervisedRunner
from dataset import FaceDataset
from const import *
import segmentation_models_pytorch as smp

img_path = img_src_dir
mask_path = mask_src_dir
val_img_path = target_dir_img
val_mask_path = target_dir_mask

train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        A.HorizontalFlip(),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

train_dataset = FaceDataset(img_path, mask_path, train_transform)
val_dataset = FaceDataset(val_img_path, val_mask_path, val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)

#model = DeepLabv3()

model = smp.Unet('resnet34', classes=1, activation=None)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.BCEWithLogitsLoss()

runner = SupervisedRunner(
    input_key='features',
    output_key='logits',
    target_key='targets',
    loss_key='loss'
)

loaders = {
    "train": train_loader,
    "valid": val_loader,
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

runner.train(
    model=model,
    criterion=loss_fn,
    optimizer=optimizer,
    loaders=loaders,
    num_epochs=10,
    callbacks=[
        dl.IOUCallback(input_key="logits", target_key="targets", threshold=0.5),
        dl.DiceCallback(input_key="logits", target_key="targets", threshold=0.5),
    ],
    logdir="./logdir",
    valid_loader="valid",
    valid_metric="loss",
    minimize_valid_metric=True,
    verbose=True,
)


def main():
    if __name__ == "__main__":
        main()
