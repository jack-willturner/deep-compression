'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
from __future__ import print_function

import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.init as init

import torchvision
import torchvision.transforms as transforms

from torch.autograd import Variable

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)



def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def get_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

def save_state(model_name, model_weights, acc):
    print('==> Saving model ...')
    state = {
            'acc': acc,
            'state_dict': model_weights.state_dict(),
            }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)

    torch.save(state, 'saved_models/ckpt'+model_name+'.t7')

def load_best(model_name, model_wts):
    filename   = 'saved_models/ckpt' + model_name + '.t7'

    checkpoint = None

    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)


    best_acc = checkpoint['acc']
    print("Loading checkpoint with best_acc: ", best_acc)

    '''
    state_dict = checkpoint['model']

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model_wts.load_state_dict(new_state_dict)
    '''
    state_dict = checkpoint['state_dict']
    model_wts.load_state_dict(state_dict)

    return model_name, model_wts, best_acc


def finetune(model, model_name, best_acc, finetuning_epochs, train_loader, test_loader, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(1, finetuning_epochs):
        train(model, epoch, optimizer, trainloader=train_loader)
        best_acc = test(model_name, model, epoch, test_loader, best_acc)
    return best_acc

# Training
def train(model, epoch,  optimizer, trainloader):
    #model_name, model = model[0], model[1]
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    print('\nEpoch: %d' % epoch)

    model.train()

    train_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()

        inputs, targets = Variable(inputs), Variable(targets)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        train_loss  += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total       += targets.size(0)
        correct     += predicted.eq(targets.data).cpu().sum()

    acc = 100.*correct/total

    print("     Accuracy: ", acc)



def test(model_name, model, epoch, testloader, best_acc):
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > best_acc:
        print('Saving..')
        save_state(model_name, model, acc)
        best_acc = acc
    return best_acc

# deep compression

def count_params(model):
    total = 0
    for param in model.parameters():
        flat = param.view(param.size(0), -1)
        flat = flat.data.cpu().numpy()
        total = total + np.count_nonzero(flat)
    return total

import numpy as np

def calculate_threshold(weights, ratio):
    return np.percentile(np.array(torch.abs(weights).cpu().numpy()), ratio)


def sparsify(model, sparsity_level=50.):
    for name, param in model.named_parameters():
        if 'weight' in name:
            threshold = calculate_threshold(param.data, sparsity_level)
            mask      = torch.gt(torch.abs(param.data), threshold).float()

            param.data = param.data * mask
    return model

def argwhere_nonzero(layer, batchnorm=False):
    indices=[]
    # for batchnorms we want to do the opposite
    if batchnorm:
        for idx,w in enumerate(layer):
            if torch.sum(torch.abs(w)).data.cpu().numpy() == 0.:
                indices.append(idx)
    else:
        for idx,w in enumerate(layer):
            if torch.sum(torch.abs(w)).data.cpu().numpy() != 0.:
                indices.append(idx)

    return indices


def prune_conv(indices, layer, follow=False):
    # follow tells us whether we need to prune input channels or output channels
    a,b,c,d = layer.weight.data.cpu().numpy().shape

    if not follow:
        # prune output channels
        layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[indices])
        if layer.bias:
            layer.bias.data   = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])
    else:
        # prune input channels - so don't touch biases because we're not changing the number of neurons/nodes/output channels
        layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:,indices])

def prune_fc(indices, channel_size, layer, follow_conv=True):
    a,b = layer.weight.data.cpu().numpy().shape
    if follow_conv:
        # if we are following a conv layer we need to expand each index by the size of the plane
        indices = [item for sublist in list((map(lambda i : np.arange((i * channel_size), (i*channel_size+channel_size)), indices))) for item in sublist]

    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:,indices])

def prune_bn(indices, layer):
    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[indices])
    layer.bias.data   = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])

    layer.running_mean = torch.from_numpy(layer.running_mean.cpu().numpy()[indices])
    layer.running_var  = torch.from_numpy(layer.running_var.cpu().numpy()[indices])

def compress_convs(model, compressed):

    ls = expand_model(model, [])

    channels = []
    nonzeros = []
    skip_connection = []

    for l1, l2 in zip(ls, ls[1:]):
        if isinstance(l1, nn.Conv2d):

            nonzeros = argwhere_nonzero(l1.weight)
            nonzeros_altered = True

            channels.append(len(nonzeros))
            channel_size = l1.kernel_size[0] * l1.kernel_size[1]
            prune_conv(nonzeros, l1)

            if isinstance(l2, nn.Conv2d):
                prune_conv(nonzeros, l2, follow=True)
            elif isinstance(l2, nn.Linear):
                prune_fc(nonzeros, channel_size, l2, follow_conv=True)
            elif isinstance(l2, nn.Sequential):
                # save for skip connection
                skip_connection = nonzeros

        elif isinstance(l1, nn.BatchNorm2d):
            # no need to append to channels since we will already have done it
            # i.e. num of channels in bn is same as num of channels in last conv layer

            assert nonzeros_altered, "batch norm layer appeared before a convolutional layer"

            l1_channels = l1.num_features

            prune_bn(nonzeros, l1)

            if isinstance(l2, nn.Conv2d):
                if (l2.in_channels < l1_channels) and (len(skip_connection) > 0): # if this is a skip connection:
                    prune_conv(skip_connection, l2, follow=True)
                elif l1_channels == l2.in_channels:
                    prune_conv(nonzeros, l2, follow=True)
            elif isinstance(l2, nn.Linear):
                prune_fc(nonzeros, channel_size, l2, follow_conv=True) # TODO fix this please

    print(channels)

    new_model = compressed(channels)

    for original, compressed in zip(expand_model(model, []), expand_model(new_model, [])):
        print("original: ", original)
        print("compressed: ", compressed)
        print("===============\n\n\n")

        if not isinstance(original, nn.Sequential):
            compressed.weight.data = original.weight.data
            if original.bias is not None:
                compressed.bias.data   = original.bias.data

    return new_model

def expand_model(model, layers=[]):
    for layer in model.children():
         if len(list(layer.children())) > 0:
             expand_model(layer, layers)
         else:
             layers.append(layer)
    return layers
