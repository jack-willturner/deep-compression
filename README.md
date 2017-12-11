# Deep Compression / Learning Weights / DSD Training 

Since the papers on deep compression are quite low on details I'll be cross-referencing these three papers:
1. Learning both weights and connections: https://arxiv.org/abs/1510.00149 
2. Deep Compression: https://arxiv.org/abs/1506.02626
3. DenseSparseDense: https://arxiv.org/abs/1607.04381

Currently just focussing on sparsity/ node pruning.

# To do:
- [x] pruning below thresholds
- [x] implement the quantization process
- [ ] extend SGD to be able to optimize with the quantized weights
- [ ] implement inference with quantized format
- [ ] huffman coding
- [ ] collate into usable modular sytem/ script
- [ ] write a notebook outlining what's happening/ how to replicate
- [ ] try to replicate results from papers
