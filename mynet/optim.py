class Optim:
    def __init__(self, parameters, lr=1):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None

    def step(self):
        for parameter in self.parameters:
            parameter.data -= parameter.grad.data * self.lr