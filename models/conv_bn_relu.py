import torch
import torch.nn as nn


class UnstructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None):
        self.mask = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.mask.weight.data = torch.ones(self.mask.weight.size())

    def update(self, new_mask):
        self.mask.weight.data  = new_mask

    def apply(self, conv, bn=None):
        conv.weight.data = torch.mul(conv.weight, self.mask.weight)


class StructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None):
        self.mask = nn.Parameter(torch.ones(planes))

    def apply(self, conv, bn):
        conv.weight.data = torch.einsum("cijk,c->cijk", conv.weight.data, self.mask)


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        relu=True,
    ):
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(planes)

        if relu:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()

        self.mask = UnstructuredMask(
            in_planes, planes, kernel_size, stride, padding, bias
        )

    def forward(self, x):
        self.mask.apply(self.conv, self.bn)

        return self.relu(self.bn(self.conv(x)))
