""" FC Layer """

import numpy as np

class FCLayer():
	def __init__(self, num_input, num_output, actFunction='relu', trainable=True):
		self.num_input = num_input
		self.num_output = num_output
		self.trainable = trainable
		self.actFunction = actFunction
		assert actFunction in ['relu', 'sigmoid']

		self.XavierInit()

		self.grad_W = np.zeros((num_input, num_output))
		self.grad_b = np.zeros((1, num_output))


	def forward(self, Input):
		self.Input = Input
		self.Output = np.dot(self.Input, self.W) + self.b
		if self.actFunction == 'relu':
			self.Output = np.maximum(self.Output, 0)
		elif self.actFunction == 'sigmoid':
			self.Output = 1 / (1 + np.exp(-self.Output))
		return self.Output

	def backward(self, delta):
		if self.actFunction == 'relu':
			delta = delta * (self.Output > 0)
		elif self.actFunction == 'sigmoid':
			delta = delta * self.Output * (1 - self.Output)
		self.grad_W = np.dot(self.Input.T, delta)
		self.grad_b = np.sum(delta, axis=0, keepdims=True)
		grad = np.dot(delta, self.W.T)
		return grad

	def XavierInit(self):
		raw_std = (2 / (self.num_input + self.num_output))**0.5
		if 'relu' == self.actFunction:
			init_std = raw_std * (2**0.5)
		elif 'sigmoid' == self.actFunction:
			init_std = raw_std
		else:
			init_std = raw_std # * 4

		self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
		self.b = np.random.normal(0, init_std, (1, self.num_output))
