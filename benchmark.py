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
from utils import *
from torch.autograd import Variable

import datetime

import pandas as pd


parser = argparse.ArgumentParser(description='CIFAR-10 Benchmark Suite')
parser.add_argument('--model', default='', help='VGG-16, ResNet-18, LeNet')
parser.add_argument('--no', action='store_true', help='Benchmark vanilla models')
parser.add_argument('--dc', action='store_true', help='Benchmark Deep Compressed models')
parser.add_argument('--bn', action='store_true', help='Benchmark BNorm Compressed models')
parser.add_argument('--l1', action='store_true', help='Benchmark Channel Lasso models')
parser.add_argument('--nvidia', action='store_true', help='Benchmark nvidia pruned models')
args = parser.parse_args()

resnet18  = ('ResNet-18', ResNet18())
mobilenet = ('MobileNet', MobileNet())
vgg_16    = ('VGG-16', VGG16())

models = {'ResNet-18': ResNet18(), 'VGG-16': VGG16(), 'MobiLeNet': MobileNet()}
#compressed_models = {'ResNet-18': ResNet18Compressed , 'VGG-16': VGG16Compressed, 'MobileNet': MobileNet()}

# If a specific model has been named then we remove everything
# else from the dictionary
model_name = args.model
if model_name != '':
    model_weights = models[model_name]
    models = {model_name: model_weights}
    #compressed_model_weights = compressed_models[model_name]
    #compressed_models = {model_name: compressed_model_weights}

train_loader, test_loader = get_data()

# benchmark vanilla models
if args.no:

    results = [] # put the results in here

    model_names = []
    accuracies  = []
    ops         = []
    mbs         = []
    inf_times   = []

    #d = {'Model':[], 'Accuracy':[], 'MAC ops':[], 'MB':[], 'Inf time':[]}
    #df = pd.DataFrame(data=d, index=[list(models.keys())])

    for model in models.items():
        acc = 0.
        model_name, model_wts = model[0], model[1]
        print("Benchmarking ", model_name)

        model_name, model_wts, best_acc = load_best(model_name, model_wts)

        tic = time.time()
        # do five times to try and remove noise
        for i in range(1,6):
            acc = test(model_name, model_wts, test_loader, best_acc)
        toc = time.time()

        total_batch_time = toc - tic
        avg_batch_time   = total_batch_time / 5


        model_names.append(model_name)
        accuracies.append(acc)
        ops.append(get_mac_ops(model_wts))
        mbs.append(get_file_size(model_name))
        inf_times.append(avg_batch_time)

    d = {'Model':model_names, 'Accuracy':accuracies, 'MAC ops':ops, 'MB':mbs, 'Inf time':inf_times}
    df = pd.DataFrame(data=d, index=[list(models.keys())])

    results_filename = 'results.csv'
    df.to_csv(results_filename)


if args.dc:
    for model in models.items():

        model_name, model_wts = model[0], model[1]
        print("Benchmarking ", model_name)

        sparsities = []
        accuracies = []
        times      = []
        params     = []


        base_model_name = model_name
        model_name, model_wts, best_acc = load_best(model_name, model_wts)
        tic = time.time()
        acc = test(model_name, model_wts, test_loader, best_acc)
        toc = time.time()

        batch_time = toc - tic

        sparsities.append(0.0)
        accuracies.append(acc)
        times.append(batch_time)
        params.append(count_params(model_wts))


        for sparsity in [50.0, 60.0, 70.0, 80.0, 90.0]:

            print("    with ", str(sparsity), " sparsity")
            model_name = model_name + str(sparsity)
            model_name, model_wts, best_acc = load_best((base_model_name + str(sparsity)), model_wts)

            tic = time.time()
            acc = test(model_name, model_wts, test_loader, best_acc)
            toc = time.time()

            batch_time = toc - tic

            sparsities.append(sparsity)
            params.append(count_params(model_wts))
            accuracies.append(acc)
            times.append(batch_time)


        d = {'Sparsity':sparsities,'Param Count':params, 'Accuracy':accuracies,'Batch Time':times}
        df = pd.DataFrame(data=d)

        results_filename =  base_model_name + 'dc_results.csv'

        df.to_csv(results_filename)
