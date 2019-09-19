'''Train base models to later be pruned'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

from models import *
from utils  import *
from tqdm   import tqdm

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model',      default='resnet18', help='resnet9/18/34/50, wrn_40_2/_16_2/_40_1')
parser.add_argument('--data_loc',   default='/disk/scratch/datasets/cifar', type=str)
parser.add_argument('--checkpoint', default='resnet18', type=str)
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')

### training specific args
parser.add_argument('--epochs',     default=200, type=int)
parser.add_argument('--lr',         default=0.1)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float, help='learning rate decay')
parser.add_argument('--weight_decay', default=0.0005, type=float)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

epoch_step = json.loads(args.epoch_step)
global error_history

models = {'resnet9'  : ResNet9(),
          'resnet18' : ResNet18(),
          'resnet34' : ResNet34(),
          'resnet50' : ResNet50(),
          'wrn_40_2' : WideResNet(40, 2),
          'wrn_16_2' : WideResNet(16, 2),
          'wrn_40_1' : WideResNet(40, 1)}

model = models[args.model]

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

trainloader, testloader = get_cifar_loaders(args.data_loc)
optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer,args.epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

error_history = []
for epoch in tqdm(range(args.epochs)):
    train(model, trainloader, criterion, optimizer)
    validate(model, epoch, testloader, criterion, checkpoint=args.checkpoint if epoch != 2 else args.checkpoint+'_init')
    scheduler.step()
