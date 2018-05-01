import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from .ConvLayer import Conv2D

import torch.nn.functional as F


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(18432, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 10),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, channels=[]):
    layers = []
    in_channels = 3
    i=0
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = Conv2D(prunable=True,in_channels=in_channels, out_channels=v, kernel_size=3, padding=2)
            if len(channels) > 0:
                conv2d = nn.Conv2d(in_channels=in_channels, out_channels=channels[i], kernel_size=3, padding=2)

            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        i += 1
    return nn.Sequential(*layers)



cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16compressed(channels, **kwargs):
    model = VGG(make_layers(cfg['D'], channels), **kwargs)
    return model

def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


'''VGG11/13/16/19 in Pytorch.
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG16(nn.Module):
    def __init__(self, ):
        super(VGG16, self).__init__()
        channels = [64,64,128,128,256,256,256,512,512,512,512,512,512]

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=2)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=2)
        self.bn2   = nn.BatchNorm2d(channels[1])
        self.pool1 = nn.MaxPool2d(2, stride=2) # TODO check this

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=2)
        self.bn3   = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=2)
        self.bn4   = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=2)
        self.bn5   = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, 3, stride=1, padding=2)
        self.bn6   = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, 3, stride=1, padding=2)
        self.bn7   = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv8 = nn.Conv2d(256, 512, 3, stride=1, padding=2)
        self.bn8   = nn.BatchNorm2d(512)
        self.conv9  = nn.Conv2d(512, 512, 3, stride=1, padding=2)
        self.bn9    = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, 3, stride=1, padding=2)
        self.bn10   = nn.BatchNorm2d(512)
        self.pool4  = nn.MaxPool2d(2, stride=2)

        self.conv11 = nn.Conv2d(512, 512, 3, stride=1, padding=2)
        self.bn11   = nn.BatchNorm2d(512)
        self.conv12 = nn.Conv2d(512, 512, 3, stride=1, padding=2)
        self.bn12   = nn.BatchNorm2d(512)
        self.conv13 = nn.Conv2d(512, 512, 3, stride=1, padding=2)
        self.bn13   = nn.BatchNorm2d(512)
        self.pool5  = nn.MaxPool2d(2, stride=2)

        self.fc1    = nn.Linear(18432, 512)
        self.fc2    = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('1.txt','\n')
        out = F.relu((self.conv2(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('2.txt','\n')
        out = self.pool1(out)

        out = F.relu((self.conv3(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('3.txt','\n')
        out = F.relu((self.conv4(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('4.txt','\n')
        out = self.pool2(out)

        out = F.relu((self.conv5(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('5.txt','\n')
        out = F.relu((self.conv6(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('6.txt','\n')
        out = F.relu((self.conv7(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('7.txt','\n')
        out = self.pool3(out)

        out = F.relu((self.conv8(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('8.txt','\n')
        out = F.relu((self.conv9(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('9.txt','\n')
        out = F.relu((self.conv10(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('10.txt','\n')
        out = self.pool4(out)

        out = F.relu((self.conv11(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('11.txt','\n')
        out = F.relu((self.conv12(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('12.txt','\n')
        out = F.relu((self.conv13(out)))
        (out.view(out.size(0), -1).data.cpu().numpy()).tofile('13.txt','\n')
        out = self.pool5(out)

        #  reshape
        out = out.view(out.size(0), -1)
        print(out.size())

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out



class VGG16Compressed(nn.Module):
    def __init__(self, channels):
        super(VGG16Compressed, self).__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, padding=3)
        self.bn1   = nn.BatchNorm2d(channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=1, padding=3)
        self.bn2   = nn.BatchNorm2d(channels[1])
        self.pool1 = nn.MaxPool2d(2, stride=2) # TODO check this

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=1, padding=2)
        self.bn3   = nn.BatchNorm2d(channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=1, padding=2)
        self.bn4   = nn.BatchNorm2d(channels[3])
        self.pool2 = nn.MaxPool2d(2, stride=2)

        self.conv5 = nn.Conv2d(channels[3], channels[4], 3, stride=1, padding=1)
        self.bn5   = nn.BatchNorm2d(channels[4])
        self.conv6 = nn.Conv2d(channels[4], channels[5], 3, stride=1, padding=1)
        self.bn6   = nn.BatchNorm2d(channels[5])
        self.conv7 = nn.Conv2d(channels[5], channels[6], 3, stride=1, padding=1)
        self.bn7   = nn.BatchNorm2d(channels[6])
        self.pool3 = nn.MaxPool2d(2, stride=2)

        self.conv8 = nn.Conv2d(channels[6], channels[7], 3, stride=1, padding=1)
        self.bn8   = nn.BatchNorm2d(channels[7])
        self.conv9  = nn.Conv2d(channels[7], channels[8], 3, stride=1, padding=1)
        self.bn9    = nn.BatchNorm2d(channels[8])
        self.conv10 = nn.Conv2d(channels[8], channels[9], 3, stride=1, padding=1)
        self.bn10   = nn.BatchNorm2d(channels[9])
        self.pool4 = nn.MaxPool2d(2, stride=2)

        self.conv11  = nn.Conv2d(channels[9], channels[10], 3, stride=1, padding=1)
        self.bn11    = nn.BatchNorm2d(channels[10])
        self.conv12  = nn.Conv2d(channels[10], channels[11], 3, stride=1, padding=1)
        self.bn12    = nn.BatchNorm2d(channels[11])
        self.conv13 = nn.Conv2d(channels[11], channels[12], 3, stride=1, padding=1)
        self.bn13   = nn.BatchNorm2d(channels[12])
        self.pool5  = nn.MaxPool2d(2, stride=2)

        self.fc1    = nn.Linear(channels[12] * 3 * 3, 512)
        self.fc2    = nn.Linear(512, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.pool1(out)

        out = F.relu(self.bn3(self.conv3(out)))
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.pool2(out)

        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = self.pool3(out)

        out = F.relu(self.bn8(self.conv8(out)))
        out = F.relu(self.bn9(self.conv9(out)))
        out = F.relu(self.bn10(self.conv10(out)))
        out = self.pool4(out)

        out = F.relu(self.bn11(self.conv11(out)))

        out = F.relu(self.bn12(self.conv12(out)))
        out = F.relu(self.bn13(self.conv13(out)))
        out = self.pool5(out)

        #  reshape
        out = out.view(out.size(0), -1)

        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out
'''
