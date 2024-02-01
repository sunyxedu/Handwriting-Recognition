""" ReLU Layer """

import numpy as np

class ReLULayer():
	def __init__(self):
		self.trainable = False

	def forward(self, Input):

		self.Input = Input
		self.Output = np.maximum(self.Input, 0)
		return self.Output

	def backward(self, delta):
		if delta is None:
			delta = 1
		if self.Input is None:
			self.Input = 0
		grad = delta * (self.Input > 0)
		return grad