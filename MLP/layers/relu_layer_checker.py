import numpy as np
from relu_layer import ReLULayer

relu_layer = ReLULayer()
input_data = np.random.randn(1, 10)
output_data = relu_layer.forward(input_data)
expected_output = np.maximum(input_data, 0)
assert np.allclose(output_data, expected_output), "Test failed!"
relu_layer = ReLULayer()
relu_layer.Input = np.array([[-1, 0, 1], [2, -3, 4]])
delta = np.ones((2, 3))
output_data = relu_layer.backward(delta)
expected_output = np.array([[0, 0, 1], [1, 0, 1]])
assert np.allclose(output_data, expected_output), "Test failed!"