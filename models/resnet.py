"""
Modified from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F

from .conv_bn_relu import ConvBNReLU

__all__ = ["ResNet9", "ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]


class Shortcut(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        expansion=1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False,
    ):
        super(Shortcut, self).__init__()
        self.conv_bn = ConvBNReLU(
            in_planes,
            expansion * planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
            relu=False,
        )

    def forward(self, x):
        return self.conv_bn(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = ConvBNReLU(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = ConvBNReLU(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, relu=False
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Shortcut(
                in_planes,
                planes,
                self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_prunable_layers(self, pruning_type="unstructured"):
        if pruning_type == "unstructured":
            return [self.conv1, self.conv2, self.shortcut.conv_bn]

        elif pruning_type == "structured":
            return [self.conv1]

        else:
            raise NotImplementedError


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        # keep a log of most recent activations to do Fisher pruning
        self.activations = []

        self.conv_bn_relu_1 = ConvBNReLU(in_planes, planes, kernel_size=1, bias=False)
        self.conv_bn_relu_2 = ConvBNReLU(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv_bn = ConvBNReLU(
            planes, self.expansion * planes, kernel_size=1, bias=False, relu=False
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Shortcut(
                in_planes,
                planes,
                self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x):
        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_relu_2(out)
        out = self.conv_bn(out)  # no relu
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    def get_prunable_layers(self, pruning_type="unstructured"):
        if pruning_type == "unstructured":
            return [
                self.conv_bn_relu_1,
                self.conv_bn_relu_2,
                self.conv_bn,
                self.shortcut.conv_bn,
            ]

        elif pruning_type == "structured":
            return [self.conv_bn_relu_1, self.conv_bn_relu_2]

        else:
            raise ValueError("Invalid type of pruning")


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv_bn_relu = ConvBNReLU(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv_bn_relu(x)

        self.activations = []

        for layer in self.layer1:
            out = layer(out)
            self.activations.append(out)

        for layer in self.layer2:
            out = layer(out)
            self.activations.append(out)

        for layer in self.layer3:
            out = layer(out)
            self.activations.append(out)

        for layer in self.layer4:
            out = layer(out)
            self.activations.append(out)

        out = F.avg_pool2d(out, 4)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []

        if pruning_type == "unstructured":
            convs.append(self.conv_bn_relu)

        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for layer in stage:
                for conv in layer.get_prunable_layers(pruning_type):
                    convs.append(conv)

        return convs


def ResNet9():
    return ResNet(BasicBlock, [1, 1, 1, 1])


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
