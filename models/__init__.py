from .resnet import *
from .wideresnet import *
from .mobilenetV1 import *
from .mobilenetV2 import *

__all__ = ["get_model"]


def get_model(model):
    if model == "resnet9":
        return ResNet9()
    elif model == "resnet18":
        return ResNet18()
    elif model == "resnet34":
        return ResNet34()
    elif model == "resnet50":
        return ResNet50()
    elif model == "wrn_40_2":
        return WideResNet(40, 2)
    elif model == "wrn_16_2":
        return WideResNet(16, 2)
    elif model == "wrn_40_1":
        return WideResNet(40, 1)
    elif model == "mobilenetV1":
        return mobilenet1()
    elif model == "mobilenetV2":
        return mobilenet2()
    else:
        raise NotImplementedError
