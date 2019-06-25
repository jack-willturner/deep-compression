# [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626)

A PyTorch implementation of [this paper](https://arxiv.org/abs/1506.02626).

I'm currently in the process of updating this to work with the latest version of PyTorch! Currently the only network type that works is ResNet - other networks coming soon. 

To run, try:
```bash
python train.py
python prune.py
```

Preliminary results on ResNet-18 look reasonable:

![alt text](./resnet-18.png)
