"""Train base models to later be pruned"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
import json
import argparse

import random
import numpy as np

from models import get_model
from utils import *
from tqdm import tqdm

################################################################## ARGUMENT PARSING

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument(
    "--model", default="resnet18", help="resnet9/18/34/50, wrn_40_2/_16_2/_40_1"
)
parser.add_argument("--data_loc", default="/disk/scratch/datasets/cifar", type=str)
parser.add_argument("--checkpoint", default=None, type=str)
parser.add_argument("--n_gpus", default=0, type=int, help="Number of GPUs to use")

### training specific args
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.1)
parser.add_argument(
    "--lr_decay_ratio", default=0.2, type=float, help="learning rate decay"
)
parser.add_argument("--weight_decay", default=0.0005, type=float)

### reproducibility
parser.add_argument("--seed", default=1, type=int)
args = parser.parse_args()

print(args.data_loc)

################################################################## REPRODUCIBILITY

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

################################################################## MODEL LOADING

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    select_devices(num_gpus_to_use=args.n_gpus)

model = get_model(args.model)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

if args.checkpoint is None:
    args.checkpoint = args.model

################################################################## TRAINING HYPERPARAMETERS

trainloader, testloader = get_cifar_loaders(args.data_loc)
optimizer = optim.SGD(
    [w for name, w in model.named_parameters() if not "mask" in name],
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.weight_decay,
)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-10)
criterion = nn.CrossEntropyLoss()

################################################################## ACTUAL TRAINING

error_history = []
for epoch in tqdm(range(args.epochs)):
    train(model, trainloader, criterion, optimizer)
    validate(
        model,
        epoch,
        testloader,
        criterion,
        checkpoint=args.checkpoint if epoch != 2 else args.checkpoint + "_init",
        seed=args.seed,
    )
    scheduler.step()
