import shutil
import glob
import os

source_path = "/DATA/kubi/Cholec80_Train/"
target_path = "/DATA/kubi/small_1000"

if not os.path.exists(target_path):
        os.mkdir(target_path)
        
images = glob.iglob(os.path.join(source_path, '*.png'))

for idx,im in enumerate(images):
    if idx < 1000:
        shutil.copy(im, os.path.join(target_path, im.split('/')[-1]))
