# -*- coding: utf-8 -*-

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

from models import *
from torch.autograd import Variable

from functools import  reduce

criterion = nn.CrossEntropyLoss()

TRAIN_BATCH_SIZE = 100  # might be worth pushing this to 128 to bit the mem better?
TEST_BATCH_SIZE = 100   # same as above


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


#_, term_width = os.popen('stty size', 'r').read().split()
#term_width = int(term_width)

#TOTAL_BAR_LENGTH = 65.
#last_time = time.time()
#begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=2)

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
    if not torch.cuda.is_available():
        checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    else:
        checkpoint = torch.load(filename)

    best_acc = checkpoint['acc']
    print("Loading checkpoint with best_acc: ", best_acc)

    state_dict = checkpoint['state_dict']
    model_wts.load_state_dict(state_dict)

    return model_name, model_wts, best_acc

# Training
def train(model, epoch, optimizer, trainloader):
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

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(model_name, model, testloader, best_acc):
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
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        save_state(model_name, model, acc)
        best_acc = acc
    print("    best acc: ", best_acc)

    return best_acc

# deep compression

def count_params(model):
    total = 0
    for param in model.parameters():
        flat = param.view(param.size(0), -1)
        flat = flat.data.cpu().numpy()
        total = total + np.count_nonzero(flat)
    return total

def get_mac_ops(model):
    total = 0.

    input_width  = 32.
    input_height = 32.

    ls = expand_model(model, []) # this seems like the most reasonable way to iterate

    for l1 in ls:
        if isinstance(l1, nn.Conv2d):

            k_w, k_h = l1.kernel_size[0], l1.kernel_size[1]
            padding_w, padding_h  = l1.padding[0], l1.padding[1]
            stride = l1.stride[0]

            mac_ops_per_kernel = (input_width + padding_w) * (input_height + padding_h) * k_w * k_h

            input_height = (input_height - k_h + (2 * padding_h) / stride) + 1
            input_width  = (input_width  - k_w + (2 * padding_w) / stride) + 1

            mac_ops = mac_ops_per_kernel * l1.out_channels
            total  += mac_ops

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
        if layer.bias is not None:
            layer.bias.data   = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])
    else:
        # prune input channels - so don't touch biases because we're not changing the number of neurons/nodes/output channels
        layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:,indices])


def prune_fc(indices, channel_size, layer, follow_conv=True):
    # a,b = layer.weight.data.cpu().numpy().shape
    if follow_conv:
        # if we are following a conv layer we need to expand each index by the size of the plane
        indices = [item for sublist in list((map(lambda i : np.arange((i * channel_size), (i*channel_size+channel_size)), indices))) for item in sublist]

    print(indices)
    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[:,indices])

def prune_bn(indices, layer):
    layer.weight.data = torch.from_numpy(layer.weight.data.cpu().numpy()[indices])
    layer.bias.data   = torch.from_numpy(layer.bias.data.cpu().numpy()[indices])

    layer.running_mean = torch.from_numpy(layer.running_mean.cpu().numpy()[indices])
    layer.running_var  = torch.from_numpy(layer.running_var.cpu().numpy()[indices])

def compute_dims(model):
    image_dims = []

    input_width  = 32.
    input_height = 32.

    ls = expand_model(model, []) # this seems like the most reasonable way to iterate

    num_input_channels = 3 # keep track of the number of channels so that if we see a decrease, we know we have hit a shortcut and can ignore it

    for l1 in ls:
        if isinstance(l1, nn.Conv2d):
            if l1.in_channels >= num_input_channels:
                k_w, k_h = l1.kernel_size[0], l1.kernel_size[1]
                padding_h, padding_w  = l1.padding[0], l1.padding[1]
                stride = l1.stride[0]

                input_height = ((input_height + 2 * padding_h - l1.dilation[0] * (k_h - 1) - 1) / stride) + 1
                input_width  = ((input_width  + 2 * padding_w - l1.dilation[1] * (k_w - 1) - 1) / stride) + 1
                assert(input_height == input_width)

                input_height = int(input_height)
                input_width  = int(input_width)

                num_input_channels = l1.out_channels
                image_dims.append(input_height)
            else:
                image_dims.append(input_height)

        elif isinstance(l1, nn.MaxPool2d):
            k_w, k_h = l1.kernel_size, l1.kernel_size
            padding_w, padding_h  = l1.padding, l1.padding
            stride = l1.stride

            input_height = ((input_height + 2 * padding_h - l1.dilation * (k_h - 1) - 1) / stride) + 1
            input_width  = ((input_width  + 2 * padding_w - l1.dilation * (k_w - 1) - 1) / stride) + 1
            assert(input_height == input_width)
            image_dims.append(int(input_height))

    return image_dims

def compress_resnet_conv(i, layers, dims):

    channels = 0 # keep track of channel reductions

    # two possible cases:
    #    conv -> bn -> conv
    #    conv -> bn -> linear
    if isinstance(layers[i], Conv2D):
        if isinstance(layers[i+1], nn.BatchNorm2d):
            nonzeros = argwhere_nonzero(layers[i].weight)
            channels = (len(nonzeros))
            prune_conv(nonzeros, layers[i])
            prune_bn(nonzeros, layers[i+1])
            if isinstance(layers[i+2], nn.Conv2d):
                # Case 1: conv -> bn -> conv
                prune_conv(nonzeros, layers[i+2], follow=True) # only prune inputs
            else:
                # Case 2: conv -> bn -> linear
                print("1: ", layers[i])
                print("2: ", layers[i+1])
                print("3: ", layers[i+2])

                channel_size = dims[i] * layers[i].out_channels
                prune_fc(nonzeros, channel_size, layers[i+3]) # +3 because +2 would hit sequential

    return channels

def compress_convs(model, compressed):

    ls = expand_model(model, [])

    channels = []
    nonzeros = []
    skip_connection = []

    for l1, l2 in zip(ls, ls[1:]):
        if isinstance(l1, Conv2D):

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

            if nonzeros_altered:

                l1_channels = l1.num_features

                prune_bn(nonzeros, l1)

                if isinstance(l2, nn.Conv2d):
                    if (l2.in_channels < l1_channels) and (len(skip_connection) > 0): # if this is a skip connection:
                        prune_conv(skip_connection, l2, follow=True)
                    elif l1_channels == l2.in_channels:
                        prune_conv(nonzeros, l2, follow=True)
                elif isinstance(l2, nn.Linear):
                    prune_fc(nonzeros, channel_size, l2, follow_conv=True) # TODO fix this please

                nonzeros_altered = False

    print(channels)

    new_model = compressed(channels)

    for original, compressed in zip(expand_model(model, []), expand_model(new_model, [])):
        print("original: ", original)
        print("compressed: ", compressed)
        print("===============\n\n\n")

        classes_to_avoid = [nn.Sequential, nn.ReLU, nn.MaxPool2d]
        has_weight = reduce((lambda b1, b2: b1 and b2), map(lambda c: not isinstance(original, c), classes_to_avoid))

        if has_weight:
            if original.weight is not None:
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

def random_sparsify(model, percentile):
    # make sure percentile is in range 0 -> 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            kernels = m.out_channels
            num_channels_to_prune = int(kernels * percentile)
            channels_to_prune = random.sample(m.weight.data.cpu().numpy(), num_channels_to_prune)
            l1.weight.data[z] = 0.

def get_file_size(model_name):
    filename = "saved_models/ckpt" + model_name +".t7"
    st = os.stat(filename)
    return st.st_size
