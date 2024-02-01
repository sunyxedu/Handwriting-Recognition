import numpy as np
EPS = 1e-11

class SoftmaxCrossEntropyLossLayer():
	def __init__(self):
		self.accu = 0.
		self.loss = np.zeros(1, dtype='f')

	def forward(self, logit, gt):

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
		probs = np.exp(self.logit) / np.sum(np.exp(self.logit), axis=-1, keepdims=True)
		grad = probs - self.gt
		return grad