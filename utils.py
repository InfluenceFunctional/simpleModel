import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda, backends
from torch.utils import data
import matplotlib.pyplot as plt
import os
import sys
import argparse


def rolling_mean(input, run):
    output = np.zeros(len(input))
    for i in range(len(output)):
        if i < run:
            output[i] = np.average(input[0:i+1])
        else:
            output[i] = np.average(input[i - run:i+1])

    return output


def getDataSize(params):
    dataset = buildDataset(params)

    return dataset.__len__()


class modelEnsemble(nn.Module): # just for evaluation of a pre-trained ensemble
    def __init__(self,models):
        super(modelEnsemble, self).__init__()
        self.models = models
        self.models = nn.ModuleList(self.models)

    def forward(self, x):
        output = []
        for i in range(len(self.models)): # get the prediction from each model
            output.append(self.models[i](x.clone()))

        output = torch.cat(output,dim=1) #
        return output # return mean and variance of the ensemble predictions


def toyFunction(x, seed, output_length = 1):
    '''
    outputs the function we will fit
    :param x:
    :return:
    '''
    np.random.seed(seed)


    # function 1 - bilinear regression
    slopes = np.random.randn(x.shape[1])  # some linear coefficients
    y = (slopes[0]*x[:,0])**1 + (slopes[1]*x[:,1])**1
    '''
    # function 2 - biquadratic regression
    slopes = np.random.randn(x.shape[1])  # some linear coefficients
    y = (slopes[0]*x[:,0])**2 + (slopes[1]*x[:,1])**2
    '''
    # function 3 - ???
    #y = slopes[0]*np.tanh(x[:,0])*x[:,1]**3 + slopes[1]*np.exp(x[:,1])*x[:,0]
    #y = -(slopes[0]*x[:,1] + slopes[1]*x[:,0])**2 + (slopes[0]*x[:,1] + slopes[1]*x[:,0])**4
    return y


class buildDataset():
    '''
    build dataset object
    '''
    def __init__(self, params):
        if params['dataset'] == 1: # train on some known function
            np.random.seed(params['dataset seed'])
            self.samples = np.random.randn(params['dataset size'],params['input length']) # random normal in 'input length' dimensions
            self.targets = toyFunction(self.samples, params['dataset seed'], params['input length'])

        '''
        # or we can preload a dataset
        dataset = np.load('datasets/' + params['dataset']+'.npy', allow_pickle=True)
        dataset = dataset.item()
        self.samples = dataset['samples']
        self.targets = dataset['scores']
        '''

        #self.samples, self.targets = shuffle(self.samples, self.targets) # shuffle, if necessary

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def getFullDataset(self):
        return self.samples, self.targets

    def getStandardization(self):
        return np.mean(self.targets), np.var(self.targets)


def getDataloaders(params): # get the dataloaders, to load the dataset in batches
    '''
    creat dataloader objects from the dataset
    :param params:
    :return:
    '''
    training_batch = params['batch size']
    dataset = buildDataset(params)  # get data
    train_size = int(0.8 * len(dataset))  # split data into training and test sets

    test_size = len(dataset) - train_size

    # construct dataloaders for inputs and targets
    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size): # take the training data from the end - we will get the newly appended datapoints this way without ever seeing the test set
        train_dataset.append(dataset[i])
    for i in range(test_size):
        test_dataset.append(dataset[i])

    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 2, pin_memory=True)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 2, pin_memory=True)

    return tr, te, dataset.__len__()


def getModelName(ensembleIndex):
    '''
    :param params: parameters of the pipeline we are training
    :return: directory label
    '''
    dirName = "estimator=" + str(ensembleIndex)

    return dirName


def get_input():
    '''
    get the command line in put for the run-num. defaulting to a new run (0)
    :return:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', type=int, default = 0)
    cmd_line_input = parser.parse_args()
    run = cmd_line_input.run_num

    return run


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
