import os
import os.path
import shutil
import random
import cv2
from pathlib import Path
from const import MASK_BASE_DIR, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, IMG_NUM, IMG_NUM_VAL


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def valid_path(path, filename):
    full_path = os.path.join(path, filename)
    return os.path.isfile(full_path)


def make_val_path(list_path, choices, target_dir):
    for idx, i in enumerate(list_path):
        if idx in choices:
            shutil.move(i, target_dir)


def make_train_mask_path(src, dst):
    for i in range(IMG_NUM):
        folder_num = i // 2000
        cur_skin = str(i).rjust(5, '0') + '_skin' + '.png'
        cur_glasses = str(i).rjust(5, '0') + '_eye_g' + '.png'
        skin_mask_path = src / str(folder_num) / cur_skin
        glasses_mask_path = src / str(folder_num) / cur_glasses
        if os.path.exists(glasses_mask_path):
            mask_skin = cv2.imread(str(skin_mask_path), cv2.IMREAD_GRAYSCALE)
            mask_glasses = cv2.imread(str(glasses_mask_path), cv2.IMREAD_GRAYSCALE)
            os.remove(str(skin_mask_path))
            new_mask = mask_skin - mask_glasses
            new_mask = new_mask.clip(min=0)
            cv2.imwrite(str(skin_mask_path), new_mask)
        Path.replace(skin_mask_path, dst / os.path.basename(skin_mask_path))


def remove_unnecessary_img(path):
    full_img_lst = os.listdir(str(path))
    full_img_lst = sorted(full_img_lst, key=lambda x: int(x.split('.')[0]))
    for i in full_img_lst:
        if IMG_NUM <= int(os.path.basename(i.split('.')[0])):
            os.remove(str(path / i))


if IMG_NUM != 30000:
    remove_unnecessary_img(TRAIN_IMG_DIR)

make_folder(str(TRAIN_MASK_DIR))
make_folder(str(VAL_IMG_DIR))
make_folder(str(VAL_MASK_DIR))

make_train_mask_path(MASK_BASE_DIR, TRAIN_MASK_DIR)

list_img = os.listdir(str(TRAIN_IMG_DIR))
list_img = sorted(list_img, key=lambda x: int(x.split('.')[0]))
files = [TRAIN_IMG_DIR / f for f in list_img if valid_path(TRAIN_IMG_DIR, f)]

list_mask = os.listdir(str(TRAIN_MASK_DIR))
list_mask = sorted(list_mask)
files_mask = [TRAIN_MASK_DIR / f for f in list_mask if valid_path(TRAIN_MASK_DIR, f)]

choice = sorted(random.sample(range(IMG_NUM), IMG_NUM_VAL))

make_val_path(files, choice, str(VAL_IMG_DIR))
make_val_path(files_mask, choice, str(VAL_MASK_DIR))
