from mynet import Value

def maximum(*X):
    X = list(X)
    while len(X) > 1:
        if not isinstance(X[0], Value): X[0] = Value(X[0])
        X[0] = X[0].max(X[1])
        X.pop(1)
    return X[0]

def ReLU(X):
    return [maximum(x, 0) for x in X]

def Sigmoid(X):
    return [1/(1+(-x).exp()) for x in X]

def Softmax(X):
    m = max(X)
    exp = [(x-m).exp() for x in X]
    norm_base = sum(exp)
    return [x / norm_base for x in exp]