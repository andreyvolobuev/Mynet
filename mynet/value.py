class Value:
	def __init__(self, data=None, parents=None, grad_fn=None):
		self.data = data
		self.parents = parents or []
		self.grad = None
		self.grad_fn = grad_fn

	def var_ops(func):
		def wrapper(*args):
			args_ = []
			for arg in args:
				if not isinstance(arg, Value):
					arg = Value(arg)
				args_.append(arg)
			return func(*args_)
		return wrapper

	@var_ops
	def __mul__(self, other):
		data = self.data * other.data
		def MulBack(grad):
			self.grad += grad * other.data
			other.grad += self * grad
		return Value(data=data, parents=[self, other], grad_fn=MulBack)

	@var_ops
	def __add__(self, other):
		data = self.data + other.data
		def AddBack(grad):
			self.grad += grad
			other.grad += grad
		return Value(data=data, parents=[self, other], grad_fn=AddBack)

	@var_ops
	def __pow__(self, other):
		data = self.data**other.data
		def PowBack(grad):
			self.grad += grad*other*self**(other-1)
			other.grad += grad*self**other
		return Value(data=data, parents=[self, other], grad_fn=PowBack)

	@var_ops
	def __gt__(self, other):
		return self.data > other.data

	@var_ops
	def __ge__(self, other):
		return self.data >= other.data

	@var_ops
	def __eq__(self, other):
		return self.data == other.data

	def __hash__(self):
		return id(self)

	@var_ops
	def __rpow__(self, other):
		return other ** self

	def __ipow__(self, other):
		return self**other

	def __sub__(self, other):
		return self + -other

	def __isub__(self, other):
		return self - other

	def __rsub__(self, other):
		return -self + other

	def __truediv__(self, other):
		return self * other**-1

	def __rtruediv__(self, other):
		return other * self**-1

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

	def root(self, other=None):
		other = other or Value(2)
		return self**(1/other)

	def exp(self):
		return 2.718281828459045**self

	def backward(self, initial=True):
		if initial:
			for var in reversed(self.deepwalk()):
				var.backward(initial=False)
		else:
			self.grad_fn(self.grad)

	def deepwalk(self):
		vars_seen = set()
		top_sort = []
		self.grad = Value(1)
		def _deepwalk(var):
			var.grad = var.grad or Value(0)
			if not var in vars_seen and var.grad_fn:
				vars_seen.add(var)
				for parent in var.parents:
					_deepwalk(parent)
				top_sort.append(var)
		_deepwalk(self)
		return top_sort

	def zero_grad(self):
		self.grad = Value(0)

	def __str__(self):
		return f'Value({self.data}{", grad=" if self.grad else ""}'\
                       f'{self.grad.data if self.grad else ""})'

	def __repr__(self):
		return str(self.data)
