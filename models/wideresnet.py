import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.conv1 = nn.Conv2d(
            in_planes,
            expansion * planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.mask1 = nn.Conv2d(
            in_planes,
            expansion * planes,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight, self.mask1.weight)
        return self.conv1(x)

    def __prune__(self, threshold):
        self.mode = "prune"
        self.mask1.weight.data = torch.mul(
            torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, mode="train"):
        super(BasicBlock, self).__init__()
        self.mode = mode
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.mask1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mask2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mask2.weight.data = torch.ones(self.mask2.weight.size())
        self.shortcut = nn.Sequential()

        self.equalInOut = in_planes == planes
        self.shortcut = (
            (not self.equalInOut)
            and Shortcut(
                in_planes,
                planes,
                self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight, self.mask1.weight)
        self.conv2.weight.data = torch.mul(self.conv2.weight, self.mask2.weight)

        if not self.equalInOut:
            x = F.relu(self.bn1(x))
        else:
            out = F.relu(self.bn1(x))

        out = self.conv2(F.relu(self.bn2(self.conv1(out if self.equalInOut else x))))
        return torch.add(x if self.equalInOut else self.shortcut(x), out)

    def __prune__(self, threshold):
        self.mode = "prune"
        self.mask1.weight.data = torch.mul(
            torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight
        )
        self.mask2.weight.data = torch.mul(
            torch.gt(torch.abs(self.conv2.weight), threshold).float(), self.mask2.weight
        )

        if isinstance(self.shortcut, Shortcut):
            self.shortcut.__prune__(threshold)


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes=10, mode="train"):
        super(WideResNet, self).__init__()
        self.in_planes = 64
        self.mode = mode

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6

        self.conv1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mask1 = nn.Conv2d(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mask1.weight.data = torch.ones(self.mask1.weight.size())
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

    def forward(self, x):
        self.conv1.weight.data = torch.mul(self.conv1.weight, self.mask1.weight)

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.linear(out)
        return out

    def __prune__(self, threshold):
        self.mode = "prune"
        self.mask1.weight.data = torch.mul(
            torch.gt(torch.abs(self.conv1.weight), threshold).float(), self.mask1.weight
        )
        layers = [self.block1, self.block2, self.block3]
        for layer in layers:
            for sub_block in layer:
                sub_block.__prune__(threshold)
