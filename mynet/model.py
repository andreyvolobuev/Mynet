import pickle
from mynet.layer import Layer
from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def _layers(self):
        return [getattr(self, l) for l in self.__dir__() if isinstance(getattr(self, l), Layer)]

    def parameters(self):
        return [p for l in self._layers() for p in l.parameters()]

    def forward(self, X):
        for layer in self._layers():
            X = layer.forward(X)
        return X

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
