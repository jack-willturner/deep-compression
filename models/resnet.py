'''
Modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Shortcut(nn.Module):
    def __init__(self, in_planes, planes, expansion=1, kernel_size=1, stride=1, bias=False, mode='train'):
        super(Shortcut, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.mask1 = nn.Conv2d(in_planes, expansion*planes, kernel_size=kernel_size, stride=stride, bias=False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.bn1 = nn.BatchNorm2d(expansion*planes)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        return self.bn1(self.conv1(x))

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode='train'):
        super(BasicBlock, self).__init__()
        self.mode  = mode
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask2.weight.data = torch.ones(self.mask2.weight.size())
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Shortcut(in_planes, planes, self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)
        self.conv2.weight.data = torch.mul(self.conv2.weight,  self.mask2.weight)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight)
        self.mask2.weight.data = torch.mul(torch.gt(torch.abs(self.conv2.weight), threshold).float(), self.mask2.weight)

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__prune__(threshold)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, mode='train'):
        super(Bottleneck, self).__init__()
        self.mode = mode
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.mask1 = nn.Parameter(torch.ones(self.conv1.weight.size()), requires_grad=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.mask2 = nn.Parameter(torch.ones(self.conv2.weight.size()), requires_grad=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.mask3 = nn.Parameter(torch.ones(self.conv3.weight.size()), requires_grad=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Shortcut(in_planes, planes, self.expansion, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight, self.mask1)
        self.conv2.weight.data = torch.mul(self.conv2.weight, self.mask2)
        self.conv3.weight.data = torch.mul(self.conv3.weight, self.mask3)

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1), requires_grad=False)
        self.mask2 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.conv2.weight), threshold).float(), self.mask2), requires_grad=False)
        self.mask3 = nn.Parameter(torch.mul(torch.gt(torch.abs(self.conv3.weight), threshold).float(), self.mask3), required_grad=False)

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__prune__(threshold)

    def __compress__(self):
        self.conv1.weight.data = torch.mul(self.conv1.weight, self.mask1)
        self.conv2.weight.data = torch.mul(self.conv2.weight, self.mask2)
        self.conv3.weight.data = torch.mul(self.conv3.weight, self.mask3)

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__compress__()

        self.mode = 'deploy'

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, mode='train'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.mode = mode
        self.conv1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask1  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight,  self.mask1.weight)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def __prune__(self, threshold):
        self.mode = 'prune'
        self.mask1.weight.data = torch.mul(torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight)
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for layer in layers:
            for sub_block in layer:
                sub_block.__prune__(threshold)

def ResNet9():
    return ResNet(BasicBlock, [1,1,1,1])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])
