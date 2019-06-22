'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

from models import *
from utils import *
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train', '-t', action='store_true')
parser.add_argument('--prune',      action='store_true')
parser.add_argument('--model',      default='resnet18', help='VGG-16, ResNet-18, LeNet')
parser.add_argument('--data_loc',   default='./data', type=str)
parser.add_argument('--checkpoint', default='resnet18', type=str)
#parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')

### training specific args
parser.add_argument('--epochs',     default=200, type=int)
parser.add_argument('--finetune_epochs', default=1)
parser.add_argument('--lr',         default=0.1)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)

args = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU
epoch_step = json.loads(args.epoch_step)
global error_history

if args.model == 'resnet18':
    model = ResNet9()

trainloader, testloader = get_cifar_loaders(args.data_loc)
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=epoch_step, gamma=args.lr_decay_ratio)
criterion = nn.CrossEntropyLoss()

if args.train:
    error_history = []
    for epoch in range(args.epochs):
        scheduler.step()
        train(model, trainloader, criterion, optimizer)
        validate(model, epoch, testloader, criterion, checkpoint=args.checkpoint)
