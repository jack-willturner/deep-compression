import torch
from torch.optim import SGD

class QuantizedSGD(SGD):
    '''
        SGD for quantized neural networks

    Performs a single optimization step, given a closure that re-evaluates the model
    and returns the loss (optional, probably not used here)
    '''
    def step(self,closure=None):
        '''
        Quantization steps:
            1. Group and sum the gradients
            2. Multiply by the learning rate
            3. Subtract from the centroids of the previous iteration
        '''

        # Don't really understand this. Copied from original SGD
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            '''
            For each layer in the model:
                1. Construct a *new* gradient matrix n x m (where n = number of centroids, and m = maximum cluster size)
                2. Iteratively add gradients to the matrix - row i represents gradients of all weights belonging to centroid[i]
                3. Deltas = Sum each row and multiply by the learning rate
                4. Substract deltas from centroids
            '''
            for idx, layer in enumerate(group['params'])

             # 1. Represent matrix as a list of tuples (row_index, gradient) where row_index = centroid
             centroid_grads = []

             # 2. Iteratively add gradients
             layer_indices = index_matrices[idx]
             for i in layer_indices.width
                for j in layer_indices.height
                    centroid_grads.append( (layer_indices[i][j],grad_input[i][j]) )

            # 3. Sum all the gradients for each centroid
            grad_sums = []
            for centroid in layer.centroids:
                # get all the gradients for that centroid
                all_centroid_gradients = list( filter( lambda (c,g): c == centroid, g))

            # 4. Calculate deltas
                delta    = sum(all_centroid_gradients) * learning_rate

            # 5. Subtract delta from centroid
                centroid = centroid - delta


'''
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum     = group['momentum']
            dampening    = group['dampening']
            nesterov     = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
'''
