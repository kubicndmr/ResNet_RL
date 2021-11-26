from data import get_train_dataset, get_valid_dataset
from utils.utils import print_log, result_path
from stopping import EarlyStoppingCallback
from matplotlib import pyplot as plt
from datetime import datetime
from trainer import Trainer

import model.resnet_torch as resnet
import numpy as np
import argparse
import torch

# get parser
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', default = None, type=str, help='Name for the training session')
parser.add_argument('-b', '--batch', default=32, type=int, help='Batch size')
parser.add_argument('-e', '--epochs', default=350, type=int, help='Number epochs')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate')
args = parser.parse_args()

# variables
session_name = args.name
epoch = args.epochs
batch_size = args.batch
learning_rate = args.learning_rate

# results folder
out_path = result_path(session_name)

# Print log
log_file = out_path + 'log.txt'

print_log('\n\tTraining started at: {} \n'.format(
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')), log_file)

print_log('Session Name: {}'.format(session_name), log_file)
print_log('Epochs: {}'.format(epoch), log_file)
print_log('Batch Size: {}'.format(batch_size), log_file)
print_log('Learning Rate: {}'.format(learning_rate), log_file)

# get datasets
trainSet = get_train_dataset()
validSet = get_valid_dataset()

trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=False)
validLoader = torch.utils.data.DataLoader(dataset=validSet, batch_size=batch_size, shuffle=False)

print_log("Training set size: {} samples".format(trainSet.__len__()), log_file)
print_log("Validation set size: {} samples".format(validSet.__len__()), log_file)

# set up your model
model = resnet.resnet50(pretrained = False, progress= True)

# add new FC layer for fine-tuning
fc  = torch.nn.Linear(1000, 7)
torch.nn.init.xavier_normal_(fc.weight)

model = torch.nn.Sequential(
    model,
    torch.nn.ReLU(),	
    fc)

# print model    
print_log(model, log_file, display = False)
print_log('\nNumber of Parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        log_file)

# loss function
criteria = torch.nn.MSELoss()

# set up optimizer 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# set up lr_scheduler
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
scheduler = None

# initialize Trainer object 
trainer = Trainer(model, criteria, optimizer, scheduler, trainLoader, 
                  validLoader, None, out_path)

# train 
res = trainer.fit(epoch)

# Log finish time
print_log('\n\tTraining finished at: {} \n'.format(
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')), log_file)
