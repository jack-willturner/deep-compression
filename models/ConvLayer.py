import torch.nn as nn

class Conv2D(nn.Conv2d):
    def __init__(self, prunable, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.prunable = prunable
