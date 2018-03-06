'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar, load_best, get_data, train, test, sparsify, count_params
from torch.autograd import Variable

global best_acc

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--deep-compress', '-d', default=False)
parser.add_argument('--train', '-t', action='store_true')
parser.add_argument('--model', '-m', default='', help='VGG-16, ResNet-18, LeNet')
args = parser.parse_args()


resnet18  = ('ResNet-18', ResNet18())
mobilenet = ('MobiLeNet', MobileNet())
vgg_16    = ('LeNet', LeNet())

models = {'ResNet-18': ResNet18(), 'LeNet':LeNet(), 'VGG-16': VGG('VGG16')}
compressed_models = {'ResNet-18': ResNet18Compressed, 'LeNet': LeNetCompressed, 'VGG-16': VGG('VGG16')}

# If a specific model has been named, then we just want to do the operations on one model
model_name = args.model
if model_name != '':
    model_weights = models[model_name]
    models = {model_name: model_weights}

train_loader, test_loader = get_data()

# assume training always done on GPU - so we don't check for CPU conversions here
def train_models():
    epochs = [(100, 0.1), (50, 0.01), (50, 0.001)]

    for model_name, model_weights in models.items():
        first_iter = True # tells us not to load from saved weights if on first iteration
        print("Training ", model_name)
        for num_epochs, learning_rate in epochs:
            for epoch in range(1, num_epochs):
                if first_iter:
                    best_acc = 0.
                else:
                    model_name, model_weights, best_acc = load_best(model_name, model_weights)

                optimizer = optim.SGD(model_weights.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

                train(model_weights, epoch, optimizer, train_loader)
                best_acc = test(model_name, model_weights, test_loader, best_acc)


def finetune(model_weights, best_acc, epochs, lr):
    optimizer = optim.SGD(model_weights.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1,epochs):
        train(model_weights, epoch, optimizer, train_loader)
        best_acc = test(model_name, model_weights, epoch, test_loader, best_acc)
    return best_acc

def deep_compress():
    for model_name, model_weights in models.items():
        base_model_name = model_name
        for sparsity in [50.,60.,70.,80.,90.]:
            # load the pretrained model
            model_name, model_weights = load_best(model_name, model_weights)
            model_name = model_name + str(sparsity)

            best_acc = 0.

            # sparsify
            model_weights = sparsify(model_weights, sparsity)

            # train with 0.01
            best_acc = finetune(model_weights, best_acc, 30, 0.01)
            # train with 0.001
            best_acc = finetune(model_weights, best_acc, 30, 0.001)

        new_model = compress_convs(model_weights, compressed_models[base_model_name])

        # finetune again - this is just to save the model
        finetune(new_model, 0., 10, 0.001)

if args.train:
    train_models()

if args.deep_compress:
    deep_compress()
    # zero out any channels that have a 0 batchnorm weight
    print("Compressing model...")

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
