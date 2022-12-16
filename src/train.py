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
from const import TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE


train_transform = A.Compose(
    [
        A.Resize(512, 512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        A.CoarseDropout(max_holes=1, max_height=0.7, max_width=0.7, min_height=0.2, min_width=0.2, mask_fill_value=0,
                        p=0.5),
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

train_dataset = FaceDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, train_transform)
val_dataset = FaceDataset(VAL_IMG_DIR, VAL_MASK_DIR, val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0,
                          pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=0, pin_memory=True)

model = DeepLabv3()

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



