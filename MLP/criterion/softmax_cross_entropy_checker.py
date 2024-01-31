import numpy as np
from softmax_cross_entropy import SoftmaxCrossEntropyLossLayer
logit = np.array([[1.0, 2.0, 3.0]])
gt = np.array([[0, 1, 0]])
loss_layer = SoftmaxCrossEntropyLossLayer()
loss = loss_layer.forward(logit, gt)
expected_loss = -np.log(np.exp(2.0) / (np.exp(1.0) + np.exp(2.0) + np.exp(3.0)))

if np.abs(loss - expected_loss) < 1e-5:
    print("Loss calculation is correct.")
else:
    print("Loss calculation is incorrect.")
    
exp_logit = np.exp(logit)
softmax = exp_logit / np.sum(exp_logit, axis=1, keepdims=True)
loss = -np.log(softmax[0, 1])
grad = softmax.copy()
grad[0, 1] -= 1 

grad_actual = loss_layer.backward()
grad_manual = grad
print("Actual Gradient:\n", grad_actual)
print("Manual Gradient:\n", grad_manual)
if np.allclose(grad_actual, grad_manual, atol=1e-5):
    print("Gradient calculation is correct.")
else:
    print("Gradient calculation is incorrect.")