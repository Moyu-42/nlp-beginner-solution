import numpy as np
from scipy.sparse import csr_matrix

class SoftmaxRegression():
    def __init__(self, args_len, targ_len, alpha=0.1, iters=1000):
        self.W = np.random.rand(args_len, targ_len)
        self.alpha = alpha
        self.iters = iters
    
    def softmax(val):
        return np.exp(val) / np.sum(np.exp(val), axis=1, keepdims=True)

    def fix(train_X, train_Y, test_X, test_Y):
        pass