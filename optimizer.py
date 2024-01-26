import numpy as np

class SGD(object):
    def __init__(self, model, learning_rate, momentum=0.0):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum

    def step(self):
        """One updating step, update weights"""

        layer = self.model
        if layer.trainable:

            ############################################################################
            # TODO: Put your code here
            # Calculate diff_W and diff_b using layer.grad_W and layer.grad_b.
            # You need to add momentum to this.

            # Weight update with momentum
            

            # # Weight update without momentum
            # layer.W += -self.learning_rate * layer.grad_W
            # layer.b += -self.learning_rate * layer.grad_b

            ############################################################################
