import random
from collections.abc import Sequence
from mynet import Value


class Neuron:
    def __init__(self, n_inputs=None, weights=None, bias=None):
        self.weights = weights or [Value(random.uniform(-1, 1)) for i in range(n_inputs)]
        self.bias = bias or Value(0)

    def parameters(self):
        """ Returns list of weights plus the bias of the neuron """
        return self.weights + [self.bias]

    def forward(self, X):
        """
        Multiply each value in the input data elementwise with each weight of the neuron,
        add the products togeather and add the neuron's bias.
        y = X*w + b
        """
        if not isinstance(X, Sequence):
            X = [X]
        if len(X) != len(self.weights):
            raise TypeError(
                "Length of the input data (%s) has to be equal "
                "to the length of the neuron's weights (%s) "
                % (len(X), len(self.weights))
            )
        result = self.bias
        for w, x in zip(self.weights, X):
            result += w * x
        return result