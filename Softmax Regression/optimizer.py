import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity_W = 0
        self.velocity_b = 0

    def step(self):
        layer = self.model
        if layer.trainable:
            self.velocity_W = self.momentum * self.velocity_W - self.learning_rate * layer.grad_W
            self.velocity_b = self.momentum * self.velocity_b - self.learning_rate * layer.grad_b
            layer.W += self.velocity_W
            layer.b += self.velocity_b