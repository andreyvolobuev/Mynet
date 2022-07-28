import math
from utils import debatch

@debatch
def ReLU(y):
    return [max(0, i) for i in y]

@debatch
def Softmax(y):
    m = max(y)
    exp = [math.e**(i-m) for i in y]
    norm_base = sum(exp)
    return [i / norm_base for i in exp]
