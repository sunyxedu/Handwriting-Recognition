import numpy as np
EPS = 1e-11

class SoftmaxCrossEntropyLoss(object):

    def __init__(self, num_input, num_output, trainable=True):
        self.num_input = num_input
        self.num_output = num_output
        self.trainable = trainable
        self.XavierInit()

    def softmax(self, numbers):
        MaxValue = np.max(numbers, axis=1, keepdims=True)
        numbers = np.exp(numbers - MaxValue)
        numbers = numbers / np.sum(numbers, axis=1, keepdims=True)
        return numbers
    
    def forward(self, Input, labels):
        logits = np.dot(Input, self.W) + self.b
        sft = self.softmax(logits)

        if sft is None:
            raise ValueError("None?!")
        
        prob = -np.log(sft[range(len(labels)), labels] + EPS)
        loss = np.mean(prob)

        predictions = np.argmax(sft, axis=1)
        acc = np.mean(predictions == labels)

        self.Input = Input
        self.labels = labels
        self.softmax_probs = prob
        return loss, acc

    def gradient_computing(self):
        size = len(self.labels)
        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

        for i in range(size):
            diff = self.softmax_probs[i] - (self.labels[i] == np.arange(self.num_output))
            self.grad_W += np.outer(self.Input[i], diff)
            self.grad_b += diff

        self.grad_W /= size
        self.grad_b /= size

    def XavierInit(self):
        raw_std = (2 / (self.num_input + self.num_output))**0.5
        init_std = raw_std * (2**0.5)
        self.W = np.random.normal(0, init_std, (self.num_input, self.num_output))
        self.b = np.random.normal(0, init_std, (1, self.num_output))
