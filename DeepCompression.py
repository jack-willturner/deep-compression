
import numpy as np

# Takes as input anything that extends nn.Module, so we can easily compress any PyTorch network
'''
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

'''
# returns the index of the closest centroid value
def closest(input, centroids):
    idx = (np.abs(centroids-input)).argmin()
    return idx


def quantize(input_matrix, k_means):
    '''
        Quantization Process
    ___________________________
    1. Cluster matrix into centroids
    2. Create an index matrix
    3. Iterate over the input matrix, putting the closest centroid index into the index matrix

    '''
    # 1. "k-means clustering" done as linear intervals
    # probably an opportunity for gain in better quantization - won't affect speed but maybe accuracy
    from sklearn import cluster
    kmeans = cluster.KMeans(n_clusters=3, n_init=4).fit(input_matrix.reshape((-1,1)))

    # 2. Create codebook vector
    centroids = kmeans.cluster_centers_

    # 3. Create index matrix
    index_matrix = np.ndarray(input_matrix.shape)

    # 4. Fill index matrix : TODO make this more numpy
    vectorized_matrix = input_matrix.reshape(1,-1)[0]

    for i,value in enumerate(vectorized_matrix):
        vectorized_matrix[i] = closest(value, centroids)

    index_matrix = vectorized_matrix.reshape(input_matrix.shape)

    return (index_matrix, centroids)




test = [[1,2,3],[4,5,6],[7,8,9]]
test = np.array(test)

print(test)
print()
indices, centroids = quantize(test, 3)
print(centroids)
print()
print(indices)
