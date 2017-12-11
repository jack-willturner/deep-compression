import torch
import numpy as np
import torch.nn as nn
from sklearn import cluster


class DeepCompression(nn.Module):
    def __init__(self, threshold=0.02, k_means=16):
        self.threshold = threshold
        self.k_means   = k_means

    # returns the index of the closest centroid value
    def closest(input, centroids):
        idx = (np.abs(centroids-input)).argmin()
        return idx

    def prune(self, model):
        for param in model.parameters():
            param = torch.gt((torch.abs(param)),self.threshold).float() # need to add "absolute"
        return model


    def construct_layer(centroids, index_matrix):
        for i in index_matrix:
            for j in index_matrix:
                index_matrix[i][j] = centroids[(index_matrix[i][j])]
        return index_matrix

    def construct_model(model, centroids, index_matrices):
        '''
        Given a model, a list of centroids and an index matrix,
        replace all of the weights in the model with the centroids
        using the index matrix

        @param model: Anything that subclasses torch.nn.Module
        @param centroids: A list of centroid vectors for each layer
        @param index_matrices: A list of index matrices for each layer
        '''

        for idx, layer in model.parameters():
            layer_ = construct_layer(centroids[idx], index_matrices[idx])
            layer  = layer_ # does this work??
        return model


    def quantize(self, model):
        '''
            Quantization Process
        ___________________________
        1. Cluster matrix into centroids
        2. Create an index matrix
        3. Iterate over the input matrix, putting the closest centroid index into the index matrix

        '''

        '''
        NOTE: Need to change k means to linear - not random - centroid initialisation
            : Probably best to implement this with Tensors instead of using numpy
        '''
        model_ = []
        for param in model.parameters():

            # 1. Clustering.  Probably done better as linear intervals - since a pruned network
            # will cluster heavily around 0
            kmeans = cluster.KMeans(n_clusters=k_means, n_init=20).fit(param.reshape((-1,1)))

            # 2. Create codebook vector
            centroids = kmeans.cluster_centers_

            # 3. Create index matrix
            index_matrix = np.ndarray(param.shape)

            # 4. Fill index matrix : TODO make this more numpy
            vectorized_matrix = param.reshape(1,-1)[0]

            for i,value in enumerate(vectorized_matrix):
                vectorized_matrix[i] = closest(value, centroids)

            index_matrix = vectorized_matrix.reshape(param.shape)

            model_.append((param, centroids))

        # need to reconstruct PyTorch module

        return model_

    '''
        Might not implement this. Helps compress network size but not really relevant
        to performance since we decompress the model before running inferences
    '''
    def huffman_encode(self, model):
        return 0

    def compress(self, model):
        # Prune and retrain
        model = prune(model)
        model = retrain_pruned_model(model, num_epochs)

        # Quantize and retrain
        model_ = quantize(model)
        model_ = construct_model(model, )
        model_ = retrain_quantized_model(model, num_epochs)

        # Huffman encode weight matrix
        # model__ = huffman_encode(model_)

        return model_
