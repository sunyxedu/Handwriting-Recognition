from euclidean_loss import EuclideanLossLayer
import numpy as np
epsilon = 1e-5
logit = np.random.randn(1, 10)
gt = np.random.randn(1, 10) 

loss_layer = EuclideanLossLayer()
loss_initial = loss_layer.forward(logit, gt)

grad_approx = np.zeros_like(logit)
for i in range(logit.shape[0]):
    for j in range(logit.shape[1]):
        logit_plus = logit.copy()
        logit_plus[i, j] += epsilon
        loss_plus = loss_layer.forward(logit_plus, gt)

        logit_minus = logit.copy()
        logit_minus[i, j] -= epsilon
        loss_minus = loss_layer.forward(logit_minus, gt)

        grad_approx[i, j] = (loss_plus - loss_minus) / (2 * epsilon)

grad_actual = loss_layer.backward()

print("Approximate Gradient:\n", grad_approx)
print("Actual Gradient:\n", grad_actual)
