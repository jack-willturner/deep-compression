import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from .conv_bn_relu import ConvBNReLU

__all__ = ["WideResNet"]


class Shortcut(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        expansion=1,
        kernel_size=1,
        stride=1,
        bias=False,
        mode="train",
    ):
        super(Shortcut, self).__init__()
        self.mode = mode
        self.conv_bn = ConvBNReLU(
            in_planes,
            expansion * planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
            relu=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_bn(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode="train"):
        super(BasicBlock, self).__init__()
        self.mode = mode

        self.conv1 = ConvBNReLU(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.conv2 = ConvBNReLU(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            relu=False,
        )

        self.shortcut = nn.Sequential()

        if in_planes != self.expansion * planes:
            self.shortcut = Shortcut(
                in_planes,
                planes,
                self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )

    def forward(self, x: Tensor) -> Tensor:
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
            raise ValueError("Invalid type of pruning")


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, mode="train"):
        super(WideResNet, self).__init__()
        self.in_planes = 64
        self.mode = mode

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6

        self.conv_bn_relu = ConvBNReLU(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        self.block1 = self._make_layer(n, nChannels[0], nChannels[1], stride=1)
        self.block2 = self._make_layer(n, nChannels[1], nChannels[2], stride=2)
        self.block3 = self._make_layer(n, nChannels[2], nChannels[3], stride=2)
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.linear = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

    def _make_layer(self, n, in_planes, out_planes, stride):
        layers = []
        for i in range(int(n)):
            layers.append(
                BasicBlock(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_bn_relu(x)
        self.activations = []

        for layer in self.block1:
            out = layer(out)
            self.activations.append(out)

        for layer in self.block2:
            out = layer(out)
            self.activations.append(out)

        for layer in self.block3:
            out = layer(out)
            self.activations.append(out)

        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)

        out = out.view(-1, self.nChannels)

        out = self.linear(out)
        return out

    def get_prunable_layers(self, pruning_type="unstructured"):
        convs = []

        if pruning_type == "unstructured":
            convs.append(self.conv_bn_relu)

        for block in [self.block1, self.block2, self.block3]:
            for layer in block:
                for conv in layer.get_prunable_layers(pruning_type):
                    convs.append(conv)

        return convs
