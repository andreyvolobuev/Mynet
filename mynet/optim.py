import math
from abc import ABC, abstractmethod


class Optim(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self):
        pass

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = None


class GradientDecent(Optim):
    def __init__(self, parameters, lr=1):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for parameter in self.parameters:
            parameter.data -= self.lr * parameter.grad.data 


class Adam(Optim):
    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_dy, self.v_dy, self.t = 0, 0, 0

    def step(self):
        self.t += 1
        for parameter in self.parameters:
            self.m_dy = self.beta1*self.m_dy + (1-self.beta1)*parameter.grad.data
            self.v_dy = self.beta2*self.v_dy + (1-self.beta2)*(parameter.grad.data**2)

            m_dy_ = self.m_dy/(1-self.beta1**self.t)
            v_dy_ = self.v_dy/(1-self.beta2**self.t)

            parameter.data = parameter.data - self.lr*(m_dy_/(math.sqrt(v_dy_)+self.epsilon))