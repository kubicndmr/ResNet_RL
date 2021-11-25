import pandas as pd
import numpy as np
import glob
import os

videos_path = "/DATA/kubi/Cholec80_1fps/videos/"
videos = os.listdir(videos_path)

trainList = []
validList = []
counter = 0

for i, v in enumerate(videos):
    if i < 41:
        print(v)
        frames = glob.iglob(os.path.join(videos_path, v, '*.jpg'))
        for f in frames:
            print(f)
            l = np.random.randint(0,7)
            trainList.append([f,l])
            if counter % 10 == 0:
                validList.append([f,l])
            counter += 1

print('Number of items in training list: ', counter)
dataList = pd.DataFrame(trainList, columns=['filename', 'label'])
dataList.to_csv("labels_train.csv")

dataList = pd.DataFrame(validList, columns=['filename', 'label'])
dataList.to_csv("labels_valid.csv")