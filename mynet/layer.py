from neuron import Neuron


class Layer:
    def __init__(self, n_inputs=None, n_outputs=None, neurons=None, activation=None):
        self.neurons = neurons or [Neuron(n_inputs) for n in range(n_outputs)]
        self.activation = activation

    def parameters(self):
        return [n.parameters() for n in self.neurons]

    def debatch(func):
        def wrapper(*args):
            if isinstance(args[-1][0], list):
                return [wrapper(*args[:-1], i) for i in args[-1]]
            return func(*args)
        return wrapper

    @debatch
    def forward(self, X):
        return self.activate([n.forward(X) for n in self.neurons])

    def activate(self, n):
        return self.activation(n) if self.activation else n

    @classmethod
    def new(cls, neurons, activation):
        return Layer(neurons=neurons, activation=activation)