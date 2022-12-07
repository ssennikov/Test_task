import torch
import torchvision.transforms.functional as F
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_segmentation_masks
from dataset import TestDataset
from torch.utils.data import DataLoader
import os
from model import DeepLabv3
from catalyst.dl import SupervisedRunner
from const import *


def get_prediction(model, runner):
    trans_img = A.Compose([A.Resize(512, 512),
                           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ToTensorV2(),
                           ])

    two_trans = A.Compose([A.Resize(512, 512),
                           ToTensorV2()])

    test_img = test_path
    test_list = os.listdir(test_img)
    test_dataset = TestDataset(test_img, img_transform=trans_img)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(loader=test_loader,
                              model=model,
                              resume=best_model_path)
    )))


    def show(imgs, filename):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            img.save(f'{filename}.jpg')

    for filename, i in enumerate(test_list):
        i = np.array(Image.open(test_img+i).convert('RGB'))
        transformed = two_trans(image=i)
        test = transformed["image"]
        preds = draw_segmentation_masks(image=test.type(torch.uint8),
                                        masks=torch.gt(torch.from_numpy(predictions[filename]), 0), alpha=0.7,
                                        colors='red')
        show(preds, filename)


#model = DeepLabv3()

model = smp.Unet('resnet34', classes=1, activation=None)

runner = SupervisedRunner(
    input_key='features',
    output_key='logits',
    target_key='targets',
    loss_key='loss'
)

get_prediction(model, runner)
