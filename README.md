# A PyTorch implementation of [this paper](https://arxiv.org/abs/1506.02626)
Keeping a notebook of all my ideas for implementing this.

I'm not sure what Han means when he says "quality parameter" for controlling the sparsity in each layer so for now what I'm doing is just choosing a percentage per layer I want to remove.

UPDATE: In the DenseSparseDense they use a Taylor expansion to approximate effect on accuracy of pruning. I'm guessing this is the quality parameter.

