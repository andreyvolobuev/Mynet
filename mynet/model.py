import pickle
from mynet import Layer
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        """ Declare layers and optimizer here """

    def _layers(self):
        """ Returns layers of the model in order of their declaration """
        return [getattr(self, l) for l in self.__dir__() if isinstance(getattr(self, l), Layer)]

    def parameters(self):
        """ Returns list of tunable parameters for each layer in the model """
        return [p for l in self._layers() for p in l.parameters()]

    def forward(self, X):
        """ Basic implementation of data forward pass. Can be redefined """
        for layer in self._layers():
            X = layer.forward(X)
        return X

    def save(self, filename):
        """ Save model to a filename """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        """ Load model from a file """
        with open(filename, 'rb') as f:
            return pickle.load(f)