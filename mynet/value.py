import math


class Value:
    def __init__(self, data=None, parents=None, grad_fn=None):
        self.data = data
        self.parents = parents
        self.grad_fn = grad_fn
        self.grad = None

    def ensure_values(func):
        """ Ensure both values used in math operations are instances of Value class """
        def wrapper(*args):
            v1, v2 = args
            if not isinstance(v2, Value): v2 = Value(v2)
            return func(v1, v2)
        return wrapper

    @ensure_values
    def __mul__(self, other):
        def MulBack(grad):
            self.grad += grad * other
            other.grad += self * grad
        return Value(data=self.data*other.data, parents=[self, other], grad_fn=MulBack)

    @ensure_values
    def __add__(self, other):
        def AddBack(grad):
            self.grad += grad
            other.grad += grad
        return Value(data=self.data+other.data, parents=[self, other], grad_fn=AddBack)

    @ensure_values
    def __pow__(self, other):
        def PowBack(grad):
            self.grad += grad * other * self ** (other - 1)
            other.grad += grad * self ** other
        return Value(data=self.data**other.data, parents=[self, other], grad_fn=PowBack)

    @ensure_values
    def max(self, other):
        def MaxBack(grad):
            self.grad += grad * (self > other)
            other.grad += grad * (other > self)
        return Value(data=max(self.data, other.data), parents=[self, other], grad_fn=MaxBack)

    def log(self):
        def LogBack(grad):
            self.grad += grad / self
        return Value(data=math.log(self.data), parents=[self], grad_fn=LogBack)

    def root(self, other=None):
        other = other or Value(2)
        return self ** (1 / other)

    def exp(self):
        return math.e ** self

    @ensure_values
    def __gt__(self, other):
        return self.data > other.data

    @ensure_values
    def __lt__(self, other):
        return self.data < other.data

    @ensure_values
    def __ge__(self, other):
        return self.data >= other.data

    @ensure_values
    def __le__(self, other):
        return self.data <= other.data

    @ensure_values
    def __eq__(self, other):
        return self.data == other.data

    @ensure_values
    def __rpow__(self, other):
        return other ** self

    def __ipow__(self, other):
        return self ** other

    def __sub__(self, other):
        return self + -other

    def __isub__(self, other):
        return self - other

    def __rsub__(self, other):
        return -self + other

    def __truediv__(self, other):
        return self * other ** -1

    def __rtruediv__(self, other):
        return other * self ** -1

    def __itruediv__(self, other):
        return self / other

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __hash__(self):
        return id(self)

    def __getstate__(self):
        """ Used by pickle to get state for serializing the value """
        return self.data

    def __setstate__(self, state):
        """ Used by pickle to set state for deserializing the value """
        self.data = state
        self.parents = None
        self.grad_fn = None
        self.grad = None

    def backward(self):
        """
        Calculate derivative of each value in the path of math operations
        that lead to the current Value
        """
        self.grad = Value(1)
        for value in reversed(self.deepwalk()):
            value.grad_fn(value.grad)

    def deepwalk(self):
        """ Returns path of math operations that lead to the Value """
        path, seen = [], set()
        def _deepwalk(value):
            value.grad = value.grad or Value(0)
            if value.grad_fn and value not in seen:
                seen.add(value)
                for parent in value.parents:
                    _deepwalk(parent)
                path.append(value)
        _deepwalk(self)
        return path

    def __str__(self):
        return f'Value({self.data}{", grad=" if self.grad else ""}'\
               f'{self.grad.data if self.grad else ""})'

    def __repr__(self):
        return str(self.data)
