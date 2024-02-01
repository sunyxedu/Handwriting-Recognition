""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.
		self.logit = None

	def forward(self, logit, gt):
		self.logit = logit
		self.gt = gt
		mse = np.sum((logit - gt) ** 2)
		self.loss = mse
		self.accu = np.mean(np.argmax(logit, axis=1) == np.argmax(gt, axis=1))

		return self.loss

	def backward(self):
		batch_size = self.logit.shape[0]
		grad = 2 * (self.logit - self.gt) / batch_size
		return grad