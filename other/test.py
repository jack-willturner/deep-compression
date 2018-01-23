import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np

from torch.autograd import Variable

import time
import os

import DeepCompression as dc
import tinynet

def sparsity_pct(model):
    running_avg = 0.0
    for param in model.parameters():
        layer = param.data.numpy()
        zeros = np.where(layer==0, 1, 0)
        num_zeros = zeros.sum()
        print(num_zeros)
        layer_avg = num_zeros / layer.size
        if (running_avg == 0):
            running_avg = layer_avg
        else:
            running_avg = (running_avg + layer_avg)/ 2
    return running_avg


# load pretrained model
print("Loading pretrained weights...")
model = tinynet.Net()
checkpoint = torch.load('tiny_net.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
print("     Sparsity ratio: " + str(sparsity_pct(model)))


print("Pruning model...")
model_ = dc.prune(model, pruning_threshold=0.02)
print("     Sparsity ratio: " + str(sparsity_pct(model)))

print("Quantizing model...")
q_model = dc.quantize(model_,8)
