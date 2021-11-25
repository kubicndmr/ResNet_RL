import os
import glob
import random
import shutil

dataset_path = "/DATA/kubi/Cholec80_Splitted/videos/"
train_path = "/DATA/kubi/Cholec80_Train/"
valid_path = "/DATA/kubi/Cholec80_Valid/"
test_path = "/DATA/kubi/Cholec80_Test/"

random.seed(271)

# arrange datasets
video_files = os.listdir(dataset_path)
video_files = [vf for vf in video_files if not vf.endswith('.txt')]
print('Number of available files: ', len(video_files))

train_videos = sorted(random.sample(video_files, 60))
print('Number of videos for training set: ', len(train_videos))

rest_files = [vf for vf in video_files if vf not in train_videos]

valid_videos = sorted(random.sample(rest_files, 10))
print('Number of videos for validation set:', len(valid_videos))

test_videos = sorted([vf for vf in rest_files if vf not in valid_videos])
print('Number of videos for test set:', len(test_videos))

def copy_dataset(source_path, target_path, source_videos):
    # create the target path and copy files with 1 fps, originally 25 fps
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    
    for vid in source_videos:
        print(os.path.join(source_path, vid))
        frames = glob.iglob(os.path.join(source_path, vid, '*.png'))
        
        for i,f in enumerate(frames):
            if i % 25 == 0:
                shutil.copy(f, os.path.join(target_path,f.split('/')[0]))


copy_dataset(dataset_path, test_path, test_videos)





