
# Takes as input anything that extends nn.Module, so we can easily compress any PyTorch network

class DeepCompression(..):
    def __init__(self, threshold=0.02):
        self.threshold = threshold

    def prune(self, model):
        for param in model.parameters():
            param = torch.gt((torch.abs(param)),self.threshold).float() # need to add "absolute"
        return model

    def quantize(self,model):
        return 0

    def huffman_encode(self, model):
        return 0

    def compress(self, model):
        model = prune(model)
        model = quantize(model)
        model = huffman_encode(model)
        return model


# returns the index of the closest centroid value
def closest(input, centroids):



def matrix_quantization(input_matrix):
    '''
        Quantization Process
    ___________________________
    1. Make a copy of the input matrix and flatten it into a vector
    2. Perform k-means clustering on the vector
    3. Create a codebook vector from the k-means centroids
    4. Create an index matrix
    5. Iterate over the input matrix, putting the closest centroid index into the index matrix

    '''

    # 1. Flatten input matrix into a vector
    vectorized_matrix = input_matrix.reshape(1,-1)[0]

    # 2. Perform k-means clustering
    sorted_vector = vectorized_matrix.sort
    range = max(sorted_vector) - min(sorted_vector)
    interval = range / len(sorted_vector)
    num_of_intervals = k_means
    centroids = np.arange(interval, num_of_intervals)

    # 3. Create codebook vector
    codebook = clustered_matrix.cluster_centers_

    # 4. Create index matrix
    index_matrix = np.ndarray(input_matrix.shape)

    # 5. Fill index matrix
