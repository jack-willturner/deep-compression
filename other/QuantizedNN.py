import torch
import torch.nn as nn
import numpy as np


class QuantizedNN(nn.module):

    def __init__(self, model_list, model):
        self.centroids, self.index_matrix = model_list
        self.model = model

    def forward(self, inputs):
        '''
            The weights of the layers are already quantized - no need to decompress
        '''
        model.forward(inputs)
