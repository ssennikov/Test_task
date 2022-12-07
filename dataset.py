from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


class FaceDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_skin.png').rjust(14, '0'))

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        augmentations = self.transform(image=image, mask=mask)
        image = augmentations['image']
        mask = augmentations['mask']

        return image, mask[None,:,:]


class TestDataset(Dataset):
    def __init__(self, img_dir, img_transform):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert('RGB'))

        augmentations = self.img_transform(image=image)
        image = augmentations['image']
        return image, 0
