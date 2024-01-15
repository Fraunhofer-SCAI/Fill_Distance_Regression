import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import QM9
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
import random
import os
import logging
import h5py
import scipy.io

epochs = 1000
batch_size = 562
# labels are scaled before training 
scaler = StandardScaler()  

# Define model 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(529, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits



loss_fn = nn.L1Loss()
MSE_loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Train function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, sample_batch in enumerate(dataloader):
        X, y = sample_batch['mordreds'].to(device), sample_batch['label'].to(device)
        y = y.unsqueeze(1)
        # Compute prediction error
        X = X.float()
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       

# Test function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()
    L1_loss = 0
    MSE_loss = 0
    max_error = torch.Tensor(np.zeros(1))
    
    with torch.no_grad():
        for batch, sample_batch in enumerate(dataloader, 1):
            X, y = sample_batch['mordreds'].to(device), sample_batch['label'].to(device)
            X = X.float()
            y = y.unsqueeze(1)
            pred = model(X)
            
            #inverse scaling is applied before computing test scores
            yr = torch.Tensor(scaler.inverse_transform(y.numpy()))
            predr = torch.Tensor(scaler.inverse_transform(pred.numpy()))

            maximum_batch_error = torch.max(torch.abs(predr-yr))
            max_error = torch.max(max_error, maximum_batch_error)
            L1_loss +=  torch.sum(torch.abs(predr - yr))
            MSE_loss += torch.sum(torch.abs(predr - yr)**2)
            
    L1_loss = L1_loss/size
    MSE_loss = MSE_loss/size
    return L1_loss, np.sqrt(MSE_loss), max_error




