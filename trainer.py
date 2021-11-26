from numpy.core.overrides import set_module
from sklearn.metrics import accuracy_score
from utils.utils import print_log

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import os

class Trainer:
    
    def __init__(self,               
                 model,                     # Model to be trained.
                 crit,                      # Loss function
                 optim = None,              # Optimiser
                 sched = None,              # Scheduler
                 train_dl = None,           # Training data set
                 val_test_dl = None,        # Validation (or test) data set
                 early_stopping_cb = None,  # Early stopping cb
                 out_path = None            # Output path
                 ):  

        self._model = model
        self._crit = crit
        self._optim = optim
        self._sched = sched
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._early_stopping_cb = early_stopping_cb
        self.out_path = out_path

        self.log_file = out_path + 'log.txt'

        if t.cuda.is_available():
            self._model = model.cuda()
            self._crit = crit.cuda()
            self.device = t.device("cuda")
        else:
            self.device = t.device("cpu")

            
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()}, self.out_path + 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))
    
    def restore_checkpoint(self, epoch_n):
        ckp = t.load(self.out_path + 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 
            'cuda' if t.cuda.is_available() else None)
        self._model.load_state_dict(ckp['state_dict'])
        
    def save_params(self, loss_train, loss_val, acc, lr):
        t.save({"loss_train": loss_train, "loss_val":loss_val, "acc":acc, "lr":lr}, self.out_path + 'epoch_params.pt')

        ## plot the results
        plt.plot(loss_train, label='loss_train')
        plt.plot(loss_val, label='loss_val')
        plt.yscale('log')
        plt.legend()
        plt.savefig(self.out_path + 'losses.png')

        plt.figure()
        plt.plot(acc, label='accuracy')
        plt.legend()
        plt.savefig(self.out_path + 'acc.png')
        plt.close('all')

    def train_step(self, x, y):
        self._optim.zero_grad()
        y_hat = self._model(x)
        y = y. squeeze()
        e = self._crit(y_hat,y)
        e.backward()
        self._optim.step()
        return e
        
    def val_test_step(self, x, y):
        with t.no_grad():
            y_hat = self._model(x)
            e = self._crit(y_hat,y.long())
            y_hat = y_hat.cpu().detach().numpy()
            return (e,np.argmax(y_hat, axis = 1))
    
    def train_epoch(self):
        self._model.train()
                
        epoch_loss = 0
        
        for i, (x_,y_) in enumerate(self._train_dl):
            print('\tProgress: {:.2f} %'.format((i+1) / self._train_dl.__len__()), 
                end='\r')

            x = x_.clone().detach().float().to(self.device)
            y = y_.clone().detach().float().to(self.device)
            epoch_loss += self.train_step(x,y)

        print('\t                            ', end = '\r')    
        return (epoch_loss / (i+1)).item()
    
    def val_test(self):
        self._model.eval()
            
        epoch_loss = 0
        true_class = 0
        set_len = 0

        for i,(x_,y_) in enumerate(self._val_test_dl):
            print('\tProgress: {:.2f} %'.format((i+1) / self._val_test_dl.__len__()), 
                end='\r')

            x = x_.clone().detach().float().to(self.device)
            y = y_.clone().detach().float().to(self.device)
                
            (e,y_hat) = self.val_test_step(x,y)
            epoch_loss += e
            y = y.cpu().numpy()
            y = np.argmax(y, axis = 1)
            true_class += np.sum(y==y_hat)
            set_len += len(y)

        print('\t                            ', end = '\r')        
        print_log('\taccuracy: {:.4f} %'.format(true_class/(set_len)), self.log_file)
        
        return [epoch_loss.item() / (i+1), true_class/(set_len)]
    
    def restore_last_session(self):
        if os.path.exists(self.out_path + 'checkpoints/'):
            print('Restoring Last Session!!!')
            c = len(os.listdir(self.out_path + 'checkpoints'))-1
            if not c == -1:
                print(c)
                self.restore_checkpoint(c)
            else:
                print('\n', 'Can\'t find any ckp file.',
                    'Starting without a pre-trained model!!')
        else:
            os.mkdir(self.out_path + 'checkpoints/')
            
    def fit(self, epochs=-1):
        assert self._early_stopping_cb is not None or epochs > 0
        
        self.restore_last_session()
                 
        train_loss = np.zeros((epochs,1))
        val_loss = np.zeros((epochs,1))
        acc = np.zeros((epochs,1))
        lr = np.zeros((epochs,1))

        best_acc = 0

        for e in range(epochs):
            # print epoch number
            print_log('\nEpoch : {}/{}'.format(e+1,epochs), self.log_file)

            # train
            train_loss[e]= self.train_epoch()
            print_log('\tt_loss:\t{}'.format(train_loss[e]), self.log_file)

            # validate
            val_loss[e], acc[e] = self.val_test()
            print_log('\tv_loss:\t{}'.format(val_loss[e]), self.log_file)            

            # scheduler step, get, display and step 
            if self._sched is not None:            
                lr[e] = self._sched.get_last_lr()
                print_log('\tl_rate:\t{}'.format(lr[e]), self.log_file)
                self._sched.step()
                
            self.save_params(train_loss, val_loss, acc, lr)
            
            if acc[e] > best_acc:
                
                # save checkpoint and params
                self.save_checkpoint(e)
               
                # store as best acc
                best_acc = acc[e]

        return [train_loss, val_loss, acc]
