""" Softmax交叉熵损失层 """

import numpy as np

# 为了防止分母为零，必要时可在分母加上一个极小项EPS
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = np.zeros(1, dtype='f')

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
		exp_logit = np.exp(logit - np.max(logit, axis=1, keepdims=True))
		softmax_probs = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)
		cross_entropy_loss = -np.sum(gt * np.log(softmax_probs + 1e-15)) / logit.shape[0]
		self.loss[0] = cross_entropy_loss
		predictions = np.argmax(softmax_probs, axis=1)
		gt_labels = np.argmax(gt, axis=1)
		correct_predictions = np.sum(predictions == gt_labels)
		self.accu = correct_predictions / logit.shape[0]
		
		return self.loss

	def backward(self):

		############################################################################
	    # TODO: 
		# 计算并返回梯度(与logit具有同样的尺寸)
	    ############################################################################
		probs = np.exp(self.logit) / np.sum(np.exp(self.logit), axis=-1, keepdims=True)
		grad = probs - self.gt
		return grad