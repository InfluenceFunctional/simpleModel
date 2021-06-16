import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.utils import data
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

class build_dataset(): # load the dataset
    def __init__(self,params):
        if params['dataset'] == 1: # aptamers fold with one-hot encoding
            np.random.seed(params['dataset seed'])
            self.samples = np.random.randn(params['dataset size'],params['input length']) # random normal in 'input length' dimensions
            self.slopes = np.random.randn(params['input length']) # some linear coefficients
            self.targets = self.samples @ np.transpose(self.slopes) # dot product

        #shuffle = np.arange(len(self.samples)) # optionally shuffle the inputs
        #np.random.shuffle(shuffle)
        #self.samples = self.samples[shuffle] #
        #self.targets = self.targets[shuffle]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]


def get_dataloaders(params): # get the dataloaders, to load the dataset in batches
    training_batch, dataset_size = params['batch_size'], params['dataset size']
    dataset = build_dataset(params)  # get data
    train_size = int(0.9 * len(dataset))  # split data into training and test sets
    #changed for 90% instead of 80%
    test_size = len(dataset) - train_size

    # construct dataloaders for inputs and targets
    train_dataset = []
    test_dataset = []

    for i in range(train_size):
        train_dataset.append(dataset[i])
    for i in range(train_size,train_size+test_size):
        test_dataset.append(dataset[i])

    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 2, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 2, pin_memory=True)

    return tr, te

def load_checkpoint(model, optimizer, dir_name, GPU, prev_epoch):
    if os.path.exists('ckpts/'+dir_name[:]):  #reload model
        checkpoint = torch.load('ckpts/' + dir_name[:])

        if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
            for i in list(checkpoint['model_state_dict']):
                checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        prev_epoch = checkpoint['epoch']

        if GPU == 1:
            model.cuda()  # move net to GPU
            for state in optimizer.state.values():  # move optimizer to GPU
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

        model.eval()
        print('Reloaded model: ', dir_name[:])
    else:
        print('New model: ', dir_name[:])

    return model, optimizer, prev_epoch


def auto_convergence(params, epoch, err_tr, err_te):
    converged = 0
    if epoch > params['average_over']:
        mean_tr = torch.mean(err_tr[-params['average_over']:])
        mean_te = torch.mean(err_te[-params['average_over']:])

        if torch.abs(mean_tr - err_tr[-params['average_over']])/mean_tr < params['train_margin']: # if we are not improving tr anymore
            converged = 1

        if torch.abs(mean_tr) < 0.0001: # set an error floor
            converged = 1

    return converged

def get_loss(train_data, params, model):
    inputs = train_data[0]
    targets = train_data[1]
    if params['GPU'] == 1:
        inputs = inputs.cuda()
        targets = targets.cuda()


    output = model(inputs.float())
    if np.ndim(targets) == 2:
        targets = targets[:,0]

    loss = F.mse_loss(output[:, 0], targets.float())  # loss function - some room to choose here!

    return loss


def rolling_mean(input, run):
    output = np.zeros(len(input))
    for i in range(len(output)):
        if i < run:
            output[i] = np.average(input[0:i+1])
        else:
            output[i] = np.average(input[i - run:i+1])

    return output

