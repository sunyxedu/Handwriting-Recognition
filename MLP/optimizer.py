import numpy as np

class SGD():
	def __init__(self, learningRate, weightDecay):
		self.learningRate = learningRate
		self.weightDecay = weightDecay
	
	def step(self, model):
		layers = model.layerList
		for layer in layers:
			if layer.trainable:
				# Update weights
				layer.W -= self.learningRate * (layer.grad_W + self.weightDecay * layer.W)
				# Update biases
				layer.b -= self.learningRate * layer.grad_b

