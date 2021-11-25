import numpy as np
import data as dataLoader
import torch
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeat):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.conv_out = nn.Conv2d(out_channels, in_channels, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.repeat = repeat
        
    def forward(self, x):
        
        residual = self.conv_in(x)
        
        for n in range(self.repeat):
            x = self.conv1(x)
            x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.batch_norm2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.batch_norm3(x)
            x = x + residual
            residual = x
            
            if n < self.repeat - 1:
                x = self.conv_out(x)
        
        return x 
    

class resnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_in = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.conv1 = nn.Conv2d(256, 128, 1, 2)
        self.conv2 = nn.Conv2d(512, 256, 1, 2)
        self.conv3 = nn.Conv2d(1024, 512, 1, 2)
        
        self.batch_norm_in = nn.BatchNorm2d(64)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(256)
        self.batch_norm3 = nn.BatchNorm2d(512)

        self.relu = nn.ReLU()

        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        
        self.resblock1 = ResBlock(64, 256, 3)
        self.resblock2 = ResBlock(128, 512, 4)
        self.resblock3 = ResBlock(256, 1024, 6)        
        self.resblock4 = ResBlock(512, 2048, 3)
        
        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048,7)
            
    def forward(self,x):
        x = self.conv_in(x)
        x = self.relu(self.batch_norm_in(x))
        x = self.max_pool(x)
        x = self.resblock1(x)
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.resblock2(x)
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.resblock3(x)
        x = self.relu(self.batch_norm3(self.conv3(x)))
        x = self.resblock4(x)
        x = self.globalavgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
