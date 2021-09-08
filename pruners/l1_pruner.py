import torch
import torch.nn as nn
import numpy as np


class L1Pruner:
    def __init__(self, pruning_type="unstructured"):
        self.pruning_type = pruning_type

    def structured_prune(self, model, prune_rate):

        # get all the prunable convolutions
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)

        # figure out the threshold of l1-norm under which channels should be turned off
        channel_norms = []
        for conv in convs:
            channel_norms.append(
                torch.sum(
                    torch.abs(conv.conv.weight.view(conv.conv.out_channels, -1)), axis=1
                )
            )

        threshold = np.percentile(channel_norms, prune_rate)

        # prune anything beneath the l1-threshold
        for conv in convs:
            channel_norms = torch.sum(
                torch.abs(conv.conv.weight.view(conv.conv.out_channels, -1)), axis=1
            )
            mask = conv.mask * (channel_norms < threshold)
            conv.mask = torch.einsum("cijk,c->cijk", conv.weight.data, mask)

    def unstructured_prune(self, model, prune_rate=50.0):

        # get all the prunable convolutions
        convs = model.get_prunable_layers(pruning_type=self.pruning_type)

        # collate all weights into a single vector so l1-threshold can be calculated
        all_weights = torch.Tensor()
        if torch.cuda.is_available():
            all_weights = all_weights.cuda()
        for conv in convs:
            all_weights = torch.cat((all_weights.view(-1), conv.conv.weight.view(-1)))
        abs_weights = torch.abs(all_weights.detach())

        threshold = np.percentile(abs_weights, prune_rate)

        # prune anything beneath l1-threshold
        for conv in model.get_prunable_layers(pruning_type=self.pruning_type):
            conv.mask.update(
                torch.mul(
                    torch.gt(torch.abs(conv.conv.weight), threshold).float(),
                    conv.mask.mask.weight,
                )
            )

    def prune(self, model, prune_rate):

        if self.pruning_type.lower() == "unstructured":
            self.unstructured_prune(model, prune_rate)

        elif self.pruning_type.lower() == "structured":
            self.structured_prune(model, prune_rate)

        else:
            raise ValueError("Invalid type of pruning")
