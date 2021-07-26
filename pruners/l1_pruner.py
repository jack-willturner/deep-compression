import torch
import numpy as np


def expand_model(model, layers=torch.Tensor()):
    for layer in model.children():
        if len(list(layer.children())) > 0:
            layers = expand_model(layer, layers)
        else:
            if isinstance(layer, nn.Conv2d) and "mask" not in layer._get_name():
                layers = torch.cat((layers.view(-1), layer.weight.view(-1)))
    return layers


class L1Pruner:
    def calculate_threshold(self, model, rate):
        empty = torch.Tensor()
        if torch.cuda.is_available():
            empty = empty.cuda()
        pre_abs = expand_model(model, empty)
        weights = torch.abs(pre_abs)

        return np.percentile(weights.detach().cpu().numpy(), rate)

    def sparsify(self, model, prune_rate=50.0):
        threshold = self.calculate_threshold(model, prune_rate)
        try:
            model.__prune__(threshold)
        except:
            model.module.__prune__(threshold)
        return model

    def prune(self, model, prune_rate):
        self.sparsify(model, prune_rate)
