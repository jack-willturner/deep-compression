import torch
import torch.nn as nn
import numpy as np


class QuantizedNN(nn.module):

    def __init__(self, model_list):
        self.centroids, self.index_matrix = model_list
