import os
import os.path
import shutil
import random
from const import *


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


def valid_path(dir_path, filename):
    full_path = os.path.join(dir_path, filename)
    return os.path.isfile(full_path)


def file_transfer(files, choises, target_dir):
    for idx, i in enumerate(files):
        if idx in choises:
            shutil.move(i, target_dir)


make_folder(mask_src_dir)

for i in range(img_num):
    folder_num = i // 2000
    filename = os.path.join(mask_folder_base, str(folder_num),
                            str(i).rjust(5, '0') + '_skin' + '.png')

    os.replace(filename, mask_src_dir + '\\' + filename.split('\\')[-1])

make_folder(target_dir_img)
make_folder(target_dir_mask)

src_files = (os.listdir(img_src_dir))
src_files = sorted(src_files, key=lambda x: int(x.split('.')[0]))
files = [os.path.join(img_src_dir, f) for f in src_files if valid_path(img_src_dir, f)]

src_files_mask = (os.listdir(mask_src_dir))
src_files_mask = sorted(src_files_mask)
files_mask = [os.path.join(mask_src_dir, f) for f in src_files_mask if valid_path(mask_src_dir, f)]

choises = sorted(random.sample(range(img_num), 1000))

file_transfer(files, choises, target_dir_img)
file_transfer(files_mask, choises, target_dir_mask)









