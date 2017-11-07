import torch
import torch.nn as nn
import numpy as np


class QuantizedAutograd(torch.autograd.Function):
