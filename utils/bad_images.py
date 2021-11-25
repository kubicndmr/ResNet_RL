from skimage.io import imread
import glob
import os


dataset_path = "/DATA/kubi/small_1000/"

images = glob.iglob(os.path.join(dataset_path, '*.png'))


for im in images:
    try:
        _ = imread(im)
    except:
        print(im)
        os.remove(im)
