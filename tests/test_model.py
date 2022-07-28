from model import Model
from layer import Layer
from optim import Optim
from loss import MSELoss
from activation import ReLU


class M(Model):
	def __init__(self):
		self.l1 = Layer(1, 3, activation=ReLU)
		self.l2 = Layer(3, 1)
		self.optim = Optim(self.parameters(), lr=0.1)


if __name__ == '__main__':
	"""
	Loading a pre-trained model.
	"""
	m = M.load('test_model_0.file')

	"""
	Dummy data that is intended to output 1 for input values of around 0.5
	and 0 for input values of around 0 and 1.
	"""
	X_train  = [[1], [0], [0.5], [0.05], [0.5], [0.45], [0.99], [0.55]]
	y_target = [[0], [0], [1.0], [0.00], [1.0], [0.99], [0.00], [1.00]]

	for i in range(1000):
		y_pred = m.forward(X_train)
		loss = MSELoss(y_pred, y_target)
		assert loss.data < 0.00095
		if i % 100 == 0:
			print(f'# EPOCH: {i}, LOSS:', loss.data)

		# loss.backward()      # turned this steps off as the model has already
		# m.optim.step()	   # pre-trained for around 2000 train-cycles
		# m.optim.zero_grad()  # and performs 'pretty well' on this dummy dataset

	# m.save(f'model_0.file')  # no need to save the model