from mynet import Neuron
from collections.abc import Sequence


class Layer:
    def __init__(self, n_inputs=None, n_outputs=None, neurons=None, activation=None):
        self.neurons = neurons or [Neuron(n_inputs) for n in range(n_outputs)]
        self.activation = activation

    def parameters(self):
        """ Returns list of tunable parameters for each neuron in the layer """
        return [p for n in self.neurons for p in n.parameters()]

    def forward(self, X):
        """ Pass the input data forward through each neuron in the layer """
        return [self._activate([n.forward(x) for n in self.neurons]) for x in X]

    def _activate(self, n):
        """ Apply activation function if it is set to the layer """
        return self.activation(n) if self.activation else n