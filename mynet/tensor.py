import random

class Tensor:
    def __init__(self, shape):
        self.data = self.create(shape)
        self.ops = []

    def create(self, shape, n=None):
        for d in reversed(shape):
            result = self.create(shape[:-1], [n or self.uniform() for i in range(d)])
            if result:
                return result
        return n

    def uniform(self):
        return random.uniform(-1, 1)

