import torch
import torchvision.transforms.functional as F
import numpy as np
import albumentations as A
import os
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader
from dataset import TestDataset
from model import DeepLabv3
from catalyst.dl import SupervisedRunner
from const import TEST_DIR, BEST_MODEL_DIR


def get_prediction(model, runner):

    trans_img = A.Compose([A.Resize(512, 512),
                           A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ToTensorV2(),
                           ])

    pred_trans = A.Compose([A.Resize(512, 512),
                            ToTensorV2(),
                            ])

    test_list = os.listdir(TEST_DIR)
    test_dataset = TestDataset(TEST_DIR, img_transform=trans_img)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(loader=test_loader,
                              model=model,
                              resume=BEST_MODEL_DIR)
    )))

    def save_predict_img(imgs, name):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
            img.save(f'{name}.jpg')

    for filename, i in enumerate(test_list):
        i = np.array(Image.open(TEST_DIR / i).convert('RGB'))
        transformed = pred_trans(image=i)
        test = transformed["image"]
        preds = draw_segmentation_masks(image=test.type(torch.uint8),
                                        masks=torch.gt(torch.from_numpy(predictions[filename]), 0), alpha=0.7,
                                        colors='red')
        save_predict_img(preds, filename)


net = DeepLabv3()

runner = SupervisedRunner(
    input_key='features',
    output_key='logits',
    target_key='targets',
    loss_key='loss'
)

get_prediction(net, runner)
