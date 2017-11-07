# Deep Compression
PyTorch Implementation of Deep Compression algorithm outlined here: https://arxiv.org/abs/1510.0014

Please note that it is currently not working - this is mostly pseudocode soon to be converted into actual Python/PyTorch.

Once it's all done I'll make a big iPython notebook explaining how the whole process works, for now the below overview will have to do.

# General Overview
In order to use this implementation of Deep Compression, your model will need to be implemented in PyTorch (for now - planning to extend/ write some NNVM bindings eventually) and subclass `torch.nn.Module`.

Deep Compression first **prunes** and retrains your model to a given threshold - we will attempt to tune the threshold until around 90% of the parameters are gone. Then the model is retrained with the pruned weight structure and passed to the quantizer.

The **quantizer** constructs a `QuantizedNN` from your model, which is essentially just your model + the centroid/index matrix representation. All of the weights in the model are replaced by the quantized weights and we use a special implementation of `stochastic gradient descent` to retrain the centroids/quantized model.

Finally, the network can be optionally compressed using huffman coding (I'll add flags to the script soon). Huffman coding is good if you don't need to immediately run your model but want to be able to fit it into a small amount of memory. To do any kind of training or inferences you'll need to decode this model format for now.    

# To do:
- [x] pruning below thresholds
- [x] implement the quantization process
- [x] extend SGD to be able to optimize with the quantized weights
- [x] implement inference with quantized format
- [ ] huffman coding
- [ ] collate into usable modular sytem/ script
- [ ] write a notebook outlining what's happening/ how to replicate
- [ ] try to replicate results from papers
