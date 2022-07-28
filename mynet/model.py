import json
from value import Value
from neuron import Neuron
from layer import Layer
from collections.abc import Sequence


class Model:
	def __init__(self, **kwargs):
		for k, v in kwargs.items():
			setattr(self, k, v)

	def _layers(self):
		return [l for l in self.__dict__.values() if isinstance(l, Layer)]

	def parameters(self):
		return [p for l in self._layers() for p in l.parameters()]

	def to_dict(self, obj_=None):
		obj = obj_ or self.__dict__
		for k, v in list(obj.items()):
			try:
				if k in ('parents', 'grad_fn', 'grad'):
					del obj[k]
				elif isinstance(v, Sequence):
					obj[k] = [self.to_dict(i.__dict__) for i in v]
				elif isinstance(v, type(None)):
					obj[k] = None
				else:
					del obj[k]
					if isinstance(v, Layer):
						k = '__LAYER__' + k
					obj[k] = self.to_dict(v.__dict__)
			except AttributeError:
				obj[k] = str(v)
		return obj

	def serialize(self, indent=None):
		return json.dumps(self.to_dict(), indent=indent)

	def save(self, filename):
		with open(filename, 'w') as f:
			f.write(self.serialize(indent=4))

	@classmethod
	def from_dict(cls, d):
		obj = {}
		for k, v in d.items():
			if '__LAYER__' in k:
				k = k.replace('__LAYER__', '')
				layer = Layer(
					neurons=[
						Neuron(
							weights=[Value(w['data']) for w in n['weights']],
							bias=Value(n['bias']['data'])
						) for n in v['neurons']
					],
					activation=None
				)
				obj[k] = layer
		obj = cls(**obj)
		return obj

	@classmethod
	def deserialize(cls, str):
		return cls.from_dict(json.loads(str))

	@classmethod
	def load(cls, filename):
		with open(filename, 'r') as f:
			return cls.deserialize(f.read())