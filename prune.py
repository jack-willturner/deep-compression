import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import os
import argparse
import random
import numpy as np

from models import get_model
from pruners import get_pruner
from utils import *
from tqdm import tqdm

################################################################## ARGUMENT PARSING
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 pruning")
    parser.add_argument(
        "--model",
        default="resnet18",
        help="resnet9, resnet18, resnet34, resnet50, wrn_40_2, wrn_16_2, wrn_40_1",
    )
    parser.add_argument("--data_loc", default="E:/Data/Datasets/cifar", type=str)
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="Pretrained model to start from"
    )
    parser.add_argument(
        "--prune_checkpoint", default=None, type=str, help="Where to save pruned models"
    )
    parser.add_argument("--n_gpus", default=0, type=int, help="Number of GPUs to use")
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="How often to save checkpoints in number of prunes (e.g. 10 = every 10 prunes)",
    )

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--cutout", action="store_true")

    ### pruning specific args
    parser.add_argument("--pruner", default="L1Pruner", type=str)
    parser.add_argument(
        "--pruning_type",
        default="structured",
        type=str,
        help="structured or unstructured",
    )
    parser.add_argument(
        "--prune_iters",
        default=100,
        help="how many times to repeat the prune->finetune process",
    )
    parser.add_argument(
        "--target_prune_rate",
        default=99,
        type=int,
        help="Percentage of parameters to prune",
    )
    parser.add_argument("--finetune_steps", default=100)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--weight_decay", default=0.0005, type=float)

    args = parser.parse_args()

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

    error_history = []

    model = get_model(args.model)

    if args.checkpoint is None:
        args.checkpoint = args.model

    args.checkpoint = args.checkpoint + "_" + str(args.seed)

    model, sd = load_model(model, args.checkpoint)

    if args.prune_checkpoint is None:
        args.prune_checkpoint = args.checkpoint + "_l1_"

    if torch.cuda.is_available():
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    ################################################################## PRUNER

    pruner = get_pruner(args.pruner, args.pruning_type)

    ################################################################## TRAINING HYPERPARAMETERS

    trainloader, testloader = get_cifar_loaders(args.data_loc, cutout=args.cutout)
    optimizer = optim.SGD(
        [w for name, w in model.named_parameters() if not "mask" in name],
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    # set the learning rate to be final LR
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-10)
    for epoch in range(sd["epoch"]):
        scheduler.step()
    for group in optimizer.param_groups:
        group["lr"] = scheduler.get_lr()[0]


    ################################################################## ACTUAL PRUNING/FINETUNING


    prune_rates = np.linspace(0, args.target_prune_rate, args.prune_iters)

    for prune_rate in tqdm(prune_rates):
        pruner.prune(model, prune_rate)

        if prune_rate % args.save_every == 0:
            checkpoint = args.prune_checkpoint + str(prune_rate)
        else:
            checkpoint = None  # don't bother saving anything

        finetune(model, trainloader, criterion, optimizer, args.finetune_steps)

        if checkpoint:
            validate(model, prune_rate, testloader, criterion, checkpoint=checkpoint)
