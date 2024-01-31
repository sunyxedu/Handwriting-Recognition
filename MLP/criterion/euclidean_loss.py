""" 欧式距离损失层 """

import numpy as np

class EuclideanLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = 0.
		self.logit = None

	def forward(self, logit, gt):
		"""
	      输入: (minibatch)
	      - logit: 最后一个全连接层的输出结果, 尺寸(batch_size, 10)
	      - gt: 真实标签, 尺寸(batch_size, 10)
	    """

		############################################################################
	    # TODO: 
		# 在minibatch内计算平均准确率和损失，分别保存在self.accu和self.loss里(将在solver.py里自动使用)
		# 只需要返回self.loss
	    ############################################################################
		self.logit = logit
		self.gt = gt
		mse = np.sum((logit - gt) ** 2)
		self.loss = mse
		self.accu = np.mean(np.argmax(logit, axis=1) == np.argmax(gt, axis=1))

		return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)
	    ############################################################################
		batch_size = self.logit.shape[0]
		grad = 2 * (self.logit - self.gt) / batch_size
		return grad