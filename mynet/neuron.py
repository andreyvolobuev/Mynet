import random
from activation import ReLU, Softmax
from utils import debatch


class Neuron:
    def __init__(self, n_inputs=None, weights=None, bias=None):
        self.bias = bias or random.uniform(-1, 1)
        self.weights = weights or [random.uniform(-1, 1) for i in range(n_inputs)]
        if n_inputs and len(self.weights) != n_inputs:
            raise ValueError(
                "Length of weights (%s) has to be equal "
                "to number of inputs. (%s) "
                % (n_inputs, len(self.weights))
            )

    @debatch
    def forward(self, X):
        if len(X) != len(self.weights):
            raise ValueError(
                "Length of input data (%s) has to be equal "
                "to length of the neuron's weights (%s) "
                % (len(X), len(self.weights))
            )
        result = self.bias
        for i, j in zip(self.weights, X):
            result += i*j
        return result
