import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class UnstructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None, groups=1):
        self.mask = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups
        )
        self.mask.weight.data = torch.ones(self.mask.weight.size())

    def update(self, new_mask):
        self.mask.weight.data = new_mask

    def apply(self, conv, bn=None):
        conv.weight.data = torch.mul(conv.weight, self.mask.weight.to(device))


class StructuredMask:
    def __init__(self, in_planes, planes, kernel_size, stride, padding, bias=None,groups=1):
        self.mask = nn.Parameter(torch.ones(planes)).to(device)

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
        groups=1,
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
            groups=groups
        )
        self.bn = nn.BatchNorm2d(planes)

        if relu:
            self.relu = nn.ReLU6(inplace=True)
        else:
            self.relu = nn.Identity()

        self.mask = UnstructuredMask(
            in_planes, planes, kernel_size, stride, padding, bias, groups
        )

    def forward(self, x):
        self.mask.apply(self.conv, self.bn)

        return self.relu(self.bn(self.conv(x)))
