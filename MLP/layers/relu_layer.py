""" ReLU激活层 """

import numpy as np

class ReLULayer():
	def __init__(self):
		"""
		ReLU激活函数: relu(x) = max(x, 0)
		"""
		self.trainable = False # 没有可训练的参数

	def forward(self, Input):

		############################################################################
	    # TODO: 
		# 对输入应用ReLU激活函数并返回结果
	    ############################################################################
		self.Input = Input
		self.Output = np.maximum(self.Input, 0)
		return self.Output

	def backward(self, delta):

		############################################################################
	    # TODO: 
		# 根据delta计算梯度
	    ############################################################################
		if delta is None:
			delta = 1
		if self.Input is None:
			self.Input = 0
		grad = delta * (self.Input > 0)
		return grad