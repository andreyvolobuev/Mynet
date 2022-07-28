from utils import debatch
from neuron import Neuron


class Layer:
    def __init__(self, n_inputs=None, n_outputs=None, neurons=None, activation=None):
        self.neurons = neurons or [Neuron(n_inputs) for n in range(n_outputs)]
        self.activation = activation

    @debatch
    def forward(self, X):
        return self.activate([n.forward(X) for n in self.neurons])

    def activate(self, n):
        return self.activation(n) if self.activation else n
