import math


class Optim:
    def zero_grad(self):
        """ Value accumulates gradients so it's important to reset them before calling step """
        for parameter in self.parameters:
            parameter.grad = None


class GradientDecent(Optim):
    def __init__(self, parameters, lr=1):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """ Simple optimizer updates the parameters by gradient times the learning rate """
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad.data 


class Adam(Optim):
    """ Original paper: https://arxiv.org/abs/1412.6980 """

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m, self.v, self.t = 0, 0, 0

    def step(self):
        self.t += 1
        for parameter in self.parameters:
            self.m = self.beta1*self.m + (1-self.beta1)*parameter.grad.data
            self.v = self.beta2*self.v + (1-self.beta2)*(parameter.grad.data**2)

            m_ = self.m/(1-self.beta1**self.t)
            v_ = self.v/(1-self.beta2**self.t)

            parameter.data = parameter.data - self.lr*(m_/(math.sqrt(v_)+self.epsilon))