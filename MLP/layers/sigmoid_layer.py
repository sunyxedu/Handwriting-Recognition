""" Sigmoid Layer """

import numpy as np

class SigmoidLayer():
	def __init__(self):
		"""
		Sigmoid激活函数: f(x) = 1/(1+exp(-x))
		"""
		self.trainable = False

	def forward(self, Input):
		############################################################################
	    # TODO: 
		# 对输入应用Sigmoid激活函数并返回结果
	    ############################################################################
		self.Input = Input
		self.Output = 1 / (1 + np.exp(-self.Input))
		return self.Output
	
	def backward(self, delta):
		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		if delta is None:
			raise ValueError("Delta is None.")

		sigmoid_output = self.forward(self.Input)
		grad = sigmoid_output * (1 - sigmoid_output)
		return delta * grad