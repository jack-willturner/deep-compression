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
from utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--model", default="resnet18", help="VGG-16, ResNet-18, LeNet")
parser.add_argument("--data_loc", default="/disk/scratch/datasets/cifar", type=str)
parser.add_argument(
    "--checkpoint", default="resnet18", type=str, help="Pretrained model to start from"
)
parser.add_argument(
    "--prune_checkpoint", default="", type=str, help="Where to save pruned models"
)
parser.add_argument("--GPU", default="0,1", type=str, help="GPU to use")
parser.add_argument(
    "--save_every",
    default=5,
    type=int,
    help="How often to save checkpoints in number of prunes (e.g. 10 = every 10 prunes)",
)
parser.add_argument("--cutout", action="store_true")

### training specific args
parser.add_argument("--finetune_steps", default=100)
parser.add_argument("--lr", default=0.001)
parser.add_argument("--weight_decay", default=0.0005, type=float)

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

error_history = []

models = {
    "resnet9": ResNet9(),
    "resnet18": ResNet18(),
    "resnet34": ResNet34(),
    "resnet50": ResNet50(),
    "wrn_40_2": WideResNet(40, 2),
    "wrn_16_2": WideResNet(16, 2),
    "wrn_40_1": WideResNet(40, 1),
}

model = models[args.model]

old_format = False
if "wrn" in args.model:
    old_format = True

model, sd = load_model(model, args.checkpoint, old_format)

if args.prune_checkpoint == "":
    prune_checkpoint = args.checkpoint + "_l1_"
else:
    prune_checkpoint = args.prune_checkpoint

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

trainloader, testloader = get_cifar_loaders(args.data_loc, cutout=args.cutout)
optimizer = optim.SGD(
    [w for name, w in model.named_parameters() if not "mask" in name],
    lr=args.lr,
    momentum=0.9,
    weight_decay=args.weight_decay,
)
criterion = nn.CrossEntropyLoss()

# set the learning rate to be 1/8th of final LR
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-10)
for epoch in range(sd["epoch"]):
    scheduler.step()
for group in optimizer.param_groups:
    group["lr"] = scheduler.get_lr()[0]

for prune_rate in tqdm(range(100)):
    model = sparsify(model, prune_rate)

    if prune_rate % args.save_every == 0:
        checkpoint = prune_checkpoint + str(prune_rate)
    else:
        checkpoint = None  # don't bother saving anything

    finetune(model, trainloader, criterion, optimizer, args.finetune_steps)

    if checkpoint:
        validate(model, prune_rate, testloader, criterion, checkpoint=checkpoint)
