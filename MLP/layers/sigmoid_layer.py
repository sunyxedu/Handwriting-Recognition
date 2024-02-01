""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		self.trainable = False

	def forward(self, Input):
		self.Input = Input
		self.Output = 1 / (1 + np.exp(-self.Input))
		return self.Output
	
	def backward(self, delta):
		if delta is None:
			raise ValueError("Delta is None.")

		sigmoid_output = self.forward(self.Input)
		grad = sigmoid_output * (1 - sigmoid_output)
		return delta * grad