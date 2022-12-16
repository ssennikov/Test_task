from pathlib import Path

ROOT_DIR = Path(__file__).resolve(strict=True).parent
SRC_DIR = ROOT_DIR / 'src'
DATA_DIR = ROOT_DIR / 'data'

MASK_BASE_DIR = DATA_DIR / 'CelebAMask-HQ' / 'CelebAMask-HQ-mask-anno'  # Folder with masks after unzip archive

TRAIN_IMG_DIR = DATA_DIR / 'CelebAMask-HQ' / 'CelebA-HQ-img'  # folder with images for training
TRAIN_MASK_DIR = DATA_DIR / 'CelebAMask-HQ' / 'CelebAMaskHQ-mask'  # folder with masks for training

VAL_IMG_DIR = DATA_DIR / 'CelebAMask-HQ' / 'CelebA-HQ-val-img'
VAL_MASK_DIR = DATA_DIR / 'CelebAMask-HQ' / 'CelebA-HQ-val-mask'

TEST_DIR = DATA_DIR / 'test'
BEST_MODEL_DIR = SRC_DIR / 'logdir' / 'checkpoints' / 'model.best.pth'

IMG_NUM = 5500
IMG_NUM_VAL = 500
BATCH_SIZE = 4
