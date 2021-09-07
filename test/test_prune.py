import torch
from models import get_model


def test_conv_bn_relu():
    from models.conv_bn_relu import ConvBNReLU

    for (Ci, Co) in [(16,16), (16,32), (32,16)]:

        convbnrelu = ConvBNReLU(Ci, Co, kernel_size=3)

        x = torch.rand((1, Ci, 32, 32))
        y = convbnrelu(x)

        assert y.size() == (1, Co, 32, 32)


def test_basic_block():
    from models.resnet import BasicBlock

    # equal in-out sizes
    for (Ci, Co) in [(16, 16), (32, 32), (64, 64)]:
        block = BasicBlock(Ci, Co)

        x = torch.rand((1, Ci, 12, 12))
        y = block(x)

        assert x.size() == y.size()

    # expanding channel dim
    for (Ci, Co) in [(16, 32), (32, 64), (64, 32)]:
        block = BasicBlock(Ci, Co)

        x = torch.rand((1, Ci, 12, 12))
        y = block(x)

        assert y.size() == (1, Co, 12, 12)

    # reducing spatial dim
    for (Ci, Co, H, W) in [(16, 32, 12, 12), (16, 32, 16, 16)]:
        block = BasicBlock(Ci, Co, stride=2)

        x = torch.rand((1, Ci, H, W))
        y = block(x)

        assert y.size() == (1, Co, H // 2, W // 2)


def test_simple_inf():
    model = get_model("resnet18")

    data = torch.rand((1, 3, 32, 32))

    y = model(data)

    assert y.size() == (1, 10)
