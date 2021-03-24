# blocks and convolution definitions
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from torch.autograd import Variable


class Conv(nn.Module):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class ConvBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        bottleneck,
        stride=1,
        kernel_size=3,
        padding=1,
        bias=False,
    ):
        super(ConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(
            in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(
            bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class GConvBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        bottleneck,
        group_split,
        stride=1,
        kernel_size=3,
        padding=1,
        bias=False,
    ):
        super(GConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(
            in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=bottleneck // group_split,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(
            bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class AConvBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        bottleneck,
        groups,
        stride=1,
        kernel_size=3,
        padding=1,
        bias=False,
    ):
        super(AConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(
            in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.conv = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(
            bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.conv(out)))
        out = self.conv1x1_up(out)
        return out


class DConvBottleneck(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        bottleneck,
        stride=1,
        kernel_size=3,
        padding=1,
        bias=False,
    ):
        super(DConvBottleneck, self).__init__()
        self.conv1x1_down = nn.Conv2d(
            in_planes, bottleneck, kernel_size=1, stride=1, padding=0, bias=bias
        )
        self.bn1 = nn.BatchNorm2d(bottleneck)
        self.convdw = nn.Conv2d(
            bottleneck,
            bottleneck,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=bottleneck,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck)
        self.conv1x1_up = nn.Conv2d(
            bottleneck, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1x1_down(x)))
        out = F.relu(self.bn2(self.convdw(out)))
        out = self.conv1x1_up(out)
        return out


class G2B2(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G2B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            group_split=2,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G4B2(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G4B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            group_split=4,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G8B2(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G8B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            group_split=8,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G16B2(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G16B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            group_split=16,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class A2B2(AConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(A2B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            groups=2,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class A4B2(AConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(A4B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            groups=4,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class A8B2(AConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(A8B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            groups=8,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class A16B2(AConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(A16B2, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 2,
            groups=16,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G2B4(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G2B4, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 4,
            group_split=2,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G4B4(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G4B4, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 4,
            group_split=4,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G8B4(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G8B4, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 4,
            group_split=8,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class G16B4(GConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(G16B4, self).__init__(
            in_planes,
            out_planes,
            bottleneck=out_planes // 4,
            group_split=16,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class ConvB2(ConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(ConvB2, self).__init__(
            in_planes,
            out_planes,
            out_planes // 2,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class ConvB4(ConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(ConvB4, self).__init__(
            in_planes,
            out_planes,
            out_planes // 4,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class ConvB8(ConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(ConvB8, self).__init__(
            in_planes,
            out_planes,
            out_planes // 8,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class ConvB16(ConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(ConvB16, self).__init__(
            in_planes,
            out_planes,
            out_planes // 16,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class Conv2x2(nn.Module):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=2, padding=1, bias=False
    ):
        super(Conv2x2, self).__init__()
        # Dilated 2x2 convs
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=2,
            stride=stride,
            padding=padding,
            bias=bias,
            dilation=2,
        )

    def forward(self, x):
        return self.conv(x)


class DConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        kernel_size=3,
        padding=1,
        bias=False,
        groups=None,
    ):
        super(DConv, self).__init__()
        self.convdw = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_planes if groups is None else groups,
        )
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv1x1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        return self.conv1x1(F.relu(self.bn(self.convdw(x))))


class DConvG2(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvG2, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=in_planes // 2,
        )


class DConvG4(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvG4, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=in_planes // 4,
        )


class DConvG8(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvG8, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=in_planes // 8,
        )


class DConvG16(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvG16, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=in_planes // 16,
        )


class DConvA2(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvA2, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=2,
        )


class DConvA4(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvA4, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=4,
        )


class DConvA8(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvA8, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=8,
        )


class DConvA16(DConv):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvA16, self).__init__(
            in_planes,
            out_planes,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            groups=16,
        )


class DConvB2(DConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvB2, self).__init__(
            in_planes,
            out_planes,
            out_planes // 2,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class DConvB4(DConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvB4, self).__init__(
            in_planes,
            out_planes,
            out_planes // 4,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class DConvB8(DConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvB8, self).__init__(
            in_planes,
            out_planes,
            out_planes // 8,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class DConvB16(DConvBottleneck):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConvB16, self).__init__(
            in_planes,
            out_planes,
            out_planes // 16,
            stride=stride,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )


class DConv3D(nn.Module):
    def __init__(
        self, in_planes, out_planes, stride=1, kernel_size=3, padding=1, bias=False
    ):
        super(DConv3D, self).__init__()
        # Separable conv approximating the 1x1 with a 3x3 conv3d
        self.convdw = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=in_planes,
        )
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv3d = nn.Conv3d(
            1, out_planes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0), bias=bias
        )

    def forward(self, x):
        o = F.relu(self.bn(self.convdw(x)))
        o = o.unsqueeze(1)
        # n, c, d, w, h = o.size()
        return self.conv3d(o).mean(2)


def block_function(blocktype):
    if blocktype == "Basic":
        block = BasicBlock
    elif blocktype == "Bottle":
        block = BottleBlock
    else:
        raise ValueError('Block "%s" not recognised' % blocktype)
    return block


def conv_function(convtype):
    if convtype == "Conv":
        conv = Conv
    elif convtype == "DConv":
        conv = DConv
    elif convtype == "DConvG2":
        conv = DConvG2
    elif convtype == "DConvG4":
        conv = DConvG4
    elif convtype == "DConvG8":
        conv = DConvG8
    elif convtype == "DConvG16":
        conv = DConvG16
    elif convtype == "DConvA2":
        conv = DConvA2
    elif convtype == "DConvA4":
        conv = DConvA4
    elif convtype == "DConvA8":
        conv = DConvA8
    elif convtype == "DConvA16":
        conv = DConvA16
    elif convtype == "Conv2x2":
        conv = Conv2x2
    elif convtype == "ConvB2":
        conv = ConvB2
    elif convtype == "ConvB4":
        conv = ConvB4
    elif convtype == "ConvB8":
        conv = ConvB8
    elif convtype == "ConvB16":
        conv = ConvB16
    elif convtype == "DConvB2":
        conv = DConvB2
    elif convtype == "DConvB4":
        conv = DConvB4
    elif convtype == "DConvB8":
        conv = DConvB8
    elif convtype == "DConvB16":
        conv = DConvB16
    elif convtype == "DConv3D":
        conv = DConv3D
    elif convtype == "G2B2":
        conv = G2B2
    elif convtype == "G4B2":
        conv = G4B2
    elif convtype == "G8B2":
        conv = G8B2
    elif convtype == "G16B2":
        conv = G16B2
    elif convtype == "G2B4":
        conv = G2B4
    elif convtype == "G4B4":
        conv = G4B4
    elif convtype == "G8B4":
        conv = G8B4
    elif convtype == "G16B4":
        conv = G16B4
    elif convtype == "A2B2":
        conv = A2B2
    elif convtype == "A4B2":
        conv = A4B2
    elif convtype == "A8B2":
        conv = A8B2
    elif convtype == "A16B2":
        conv = A16B2
    elif convtype == "ACDC":
        conv = ACDC
    else:
        raise ValueError('Conv "%s" not recognised' % convtype)
    return conv


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.stride = stride
        self.convtype = conv

    def __del__(self):
        b1 = [self.bn1, self.conv1, self.bn2, self.conv2, self.relu1, self.relu2]
        b2 = [self.droprate, self.equalInOut, self.convShortcut]

        for b in [b1, b2]:
            del b

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MaskBlock(nn.Module):
    expansion = 1
    """Wrap-round block for resnets. Doesnt incorporate dropout yet."""
    # conv type doesn't matter here
    def __init__(
        self, in_channels, out_channels, stride=1, dropRate=0.0, conv=Conv, bottle=False
    ):

        super(MaskBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.droprate = dropRate
        self.equalInOut = in_channels == out_channels
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

        self.activation = Identity()
        self.handle = self.activation.register_backward_hook(
            self._fisher
        )  ##### THIS LINE OF CODE IS BREAKING EVERYTHING

        self.register_buffer("mask", None)

        self.input_spatial_dims = None
        self.input_shape = None
        self.output_shape = None
        self.flops = None
        self.params = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.got_shapes = False

        # Fisher method is called on backward passes
        self.running_fisher = 0
        self.convtype = conv

    def __del__(self):
        self.handle.remove()

    def forward(self, x):
        self.input_spatial_dims = x.size()
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        if self.mask is not None:
            out = out * self.mask[None, :, None, None]
        else:
            self._create_mask(x, out)

        out = self.activation(out)

        self.act = out
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def _create_mask(self, x, out):
        """This takes an activation to generate the exact mask required. It also records input and output shapes
        for posterity."""
        self.mask = x.new_ones(out.shape[1])
        self.input_shape = x.size()
        self.output_shape = out.size()

    def _fisher(self, blargh, blergh, grad_output):
        act = self.act.detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher += del_k

    def reset_fisher(self):
        self.running_fisher = 0 * self.running_fisher

    def update(self, previous_mask):
        return None

    def get_conv_sizes(self):

        sizes = []

        for c in [self.conv1, self.conv2]:
            if isinstance(c, DConv):
                sizes.append(c.conv1x1.weight.size())
            else:
                sizes.append(c.conv.weight.size())
        return sizes[0], sizes[1]

    def cost(self):

        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size, conv2_size = self.get_conv_sizes()

        self.params = (
            in_channels * middle_channels * conv1_size[2] * conv1_size[3]
            + middle_channels * out_channels * conv2_size[2] * conv2_size[3]
        )
        self.flops = (
            self.output_shape[2]
            * self.output_shape[3]
            * conv1_size[2]
            * conv1_size[3]
            * in_channels
            * middle_channels
            + self.output_shape[2]
            * self.output_shape[3]
            * conv2_size[2]
            * conv2_size[3]
            * middle_channels
            * out_channels
        )

        self.flops_vector = self.mask * (self.flops / self.out_channels)

    def compress_weights(self):

        middle_dim = int(self.mask.sum().item())

        if middle_dim is not 0:

            conv1 = nn.Conv2d(
                self.in_channels,
                middle_dim,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                bias=False,
            )
            conv1.weight = nn.Parameter(self.conv1.weight[self.mask == 1, :, :, :])

            # Batch norm 2 changes
            bn2 = nn.BatchNorm2d(middle_dim)
            bn2.weight = nn.Parameter(self.bn2.weight[self.mask == 1])
            bn2.bias = nn.Parameter(self.bn2.bias[self.mask == 1])
            bn2.running_mean = self.bn2.running_mean[self.mask == 1]
            bn2.running_var = self.bn2.running_var[self.mask == 1]

            conv2 = nn.Conv2d(
                middle_dim,
                self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            conv2.weight = nn.Parameter(self.conv2.weight[:, self.mask == 1, :, :])

        if middle_dim is 0:
            conv1 = Zero()
            bn2 = Zero()
            conv2 = ZeroMake(channels=self.out_channels, spatial=self.stride)

        self.conv1 = conv1
        self.conv2 = conv2
        self.bn2 = bn2

        if middle_dim is not 0:
            self.mask = torch.ones(middle_dim)
        else:
            self.mask = torch.ones(1)


class MaskBottleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):
        super(MaskBottleBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

        self.convtype = conv

        self.activation = Identity()
        self.handle = self.activation.register_backward_hook(self._fisher)

        self.register_buffer("mask", None)

        self.input_spatial_dims = None
        self.input_shape = None
        self.output_shape = None
        self.flops = None
        self.params = None
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.stride = stride
        self.got_shapes = False
        self.running_fisher = 0

    def forward(self, x):
        self.input_spatial_dims = x.size()
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out if self.equalInOut else x)

        if self.mask is not None:
            out = out * self.mask[None, :, None, None]
        else:
            self._create_mask(x, out)

        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.activation(out)
        self.act = out
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

    def _create_mask(self, x, out):
        """This takes an activation to generate the exact mask required. It also records input and output shapes
        for posterity."""
        self.mask = x.new_ones(out.shape[1])
        self.input_shape = x.size()
        self.output_shape = out.size()

    def _fisher(self, blargh, blergh, grad_output):
        act = self.act.detach()
        grad = grad_output[0].detach()

        g_nk = (act * grad).sum(-1).sum(-1)
        del_k = g_nk.pow(2).mean(0).mul(0.5)
        self.running_fisher += del_k

    def reset_fisher(self):
        self.running_fisher = 0 * self.running_fisher

    def update(self, previous_mask):
        return None

    def get_conv_sizes(self):

        size = 0.0
        if isinstance(self.conv1, Conv):
            size = self.conv1.conv.weight.size()
        elif isinstance(self.conv1, DConv):
            size = self.conv1.convdw.weight.size()
        else:
            size = self.conv1.conv1x1_up.weight.size()
        return size

    def cost(self):

        in_channels = self.in_channels
        out_channels = self.out_channels
        middle_channels = int(self.mask.sum().item())

        conv1_size = self.get_conv_sizes()

        self.params = (
            in_channels * middle_channels * conv1_size[2] * conv1_size[3]
            + middle_channels * out_channels
        )
        self.flops = (
            self.output_shape[2]
            * self.output_shape[3]
            * conv1_size[2]
            * conv1_size[3]
            * in_channels
            * middle_channels
        )

        self.flops_vector = self.mask * (self.flops / self.out_channels)


class BottleBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, conv=Conv, xy=None):
        super(BottleBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.stride = stride
        self.convtype = conv

    def __del__(self):
        b1 = [self.bn1, self.conv1]
        b2 = [self.droprate, self.equalInOut, self.convShortcut]

        for b in [b1, b2]:
            del b

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = out
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


get_block_type = {
    Conv: BasicBlock,
    DConvA2: BasicBlock,
    DConvA4: BasicBlock,
    DConvA8: BasicBlock,
    DConvA16: BasicBlock,
    DConvG16: BasicBlock,
    DConvG8: BasicBlock,
    DConvG4: BasicBlock,
    DConvG2: BasicBlock,
    DConv: BasicBlock,
    ConvB2: BottleBlock,
    ConvB4: BottleBlock,
    A2B2: BottleBlock,
    A4B2: BottleBlock,
    A8B2: BottleBlock,
    A16B2: BottleBlock,
    G16B2: BottleBlock,
    G8B2: BottleBlock,
    G4B2: BottleBlock,
    G2B2: BottleBlock,
    G4B4: BottleBlock,
    G2B4: BottleBlock,
    G8B4: BottleBlock,
    DConvB4: BottleBlock,
}


def update_block(index, model, new_conv, mask=True):
    i = 0
    basic_or_bottle = get_block_type[new_conv]

    for m in model.modules():
        if isinstance(m, NetworkBlock):
            for j, sub_sub_block in enumerate(m.layer):
                if index == i:
                    if basic_or_bottle == BottleBlock:
                        if mask:
                            m.layer[j] = MaskBottleBlock(
                                sub_sub_block.in_channels,
                                sub_sub_block.out_channels,
                                conv=new_conv,
                                stride=sub_sub_block.stride,
                            ).cuda()
                        else:
                            m.layer[j] = BottleBlock(
                                sub_sub_block.in_channels,
                                sub_sub_block.out_channels,
                                conv=new_conv,
                                stride=sub_sub_block.stride,
                            ).cuda()
                    else:
                        if mask:
                            m.layer[j] = MaskBlock(
                                sub_sub_block.in_channels,
                                sub_sub_block.out_channels,
                                conv=new_conv,
                                stride=sub_sub_block.stride,
                            ).cuda()
                        else:
                            m.layer[j] = BasicBlock(
                                sub_sub_block.in_channels,
                                sub_sub_block.out_channels,
                                conv=new_conv,
                                stride=sub_sub_block.stride,
                            ).cuda()
                i = i + 1
    model.convs[index] = new_conv

    return model


class NetworkBlock(nn.Module):
    def __init__(
        self,
        nb_layers,
        in_planes,
        out_planes,
        stride,
        dropRate=0.0,
        convs=[Conv, Conv],
        masked=False,
    ):
        super(NetworkBlock, self).__init__()
        blocks = [get_block_type[conv] for conv in convs]
        if get_block_type[convs[0]] == BasicBlock:
            block = MaskBlock
        else:
            block = MaskBottleBlock

        if masked:
            blocks = [block for conv in convs]

        self.layer = self._make_layer(
            blocks, in_planes, out_planes, nb_layers, stride, dropRate, convs
        )
        self.masked = masked
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.conv = convs[0]

    def _make_layer(
        self, blocks, in_planes, out_planes, nb_layers, stride, dropRate, convs
    ):
        layers = []
        for i in range(nb_layers):
            layers.append(
                blocks[i](
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                    convs[i],
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        self.input_spatial_dims = (
            x.size()
        )  # register this to make flop calculations easy
        return self.layer(x)
