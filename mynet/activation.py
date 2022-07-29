import math

def ReLU(X):
    return [x.relu() for x in X]

def Sigmoid(X):
    return [1/(1+(-x).exp()) for x in X]

def Softmax(X):
    m = max(X)
    exp = [(x-m).exp() for x in X]
    norm_base = sum(exp)
    return [x / norm_base for x in exp]
