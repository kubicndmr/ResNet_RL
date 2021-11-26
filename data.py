from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.34918428, 0.22890709, 0.22321491]
train_std = [0.06838285, 0.06596467, 0.06959078]

class ResNetDataset(Dataset):
    def __init__(self,mode,csv_path,transformer=tv.transforms.Compose([tv.transforms.ToTensor()])):
        self.mode = mode
        self.tr = transformer
        self.aug_factor = 1
        
        self.get_data(csv_path)
        
    def __getitem__(self,index):
        data = self.data[index-1]
        
        im = imread(data[0])
        im = self.tr(im)
        
        label = torch.zeros(7)
        label[data[1]] = 1
        
        return (im,label)
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self,path):
        dataList = pd.read_csv(path).values.tolist()
        
        Data = []
        
        for data in dataList:
            filename = data[1]
            label = data[2]
            Data.append((filename,label))
            
        self.data = Data        
        
    def pos_weight(self):
        positives = np.empty((1,2))
        
        for d in self.data:
            labels = np.array(d[1])
            labels = np.expand_dims(labels,0)
            positives += labels
        
        negatives = np.array([len(self.data)-positives[0,0],len(self.data)-positives[0,1]])
        return negatives/positives
            
    
def get_train_dataset():
    ToTensor = tv.transforms.ToTensor()
    Normalize = tv.transforms.Normalize(train_mean,train_std)
    return ResNetDataset('train','labels_train.csv',tv.transforms.Compose([ToTensor, Normalize]))

def get_valid_dataset():
    ToTensor = tv.transforms.ToTensor()
    Normalize = tv.transforms.Normalize(train_mean,train_std)
    return ResNetDataset('train','labels_valid.csv',tv.transforms.Compose([ToTensor, Normalize]))
