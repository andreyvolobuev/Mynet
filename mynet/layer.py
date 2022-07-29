from mynet.neuron import Neuron
from collections.abc import Sequence


class Layer:
    def __init__(self, n_inputs=None, n_outputs=None, neurons=None, activation=None):
        self.neurons = neurons or [Neuron(n_inputs) for n in range(n_outputs)]
        self.activation = activation

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def forward(self, X):
        return [self.activate([n.forward(x) for n in self.neurons]) for x in X]

    def activate(self, n):
        return self.activation(n) if self.activation else n