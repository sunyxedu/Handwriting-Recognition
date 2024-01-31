import numpy as np
from sigmoid_layer import SigmoidLayer

def test_sigmoid_layer():
    sigmoid_layer = SigmoidLayer()
    input_data = np.random.randn(10, 10)
    output_data = sigmoid_layer.forward(input_data)
    expected_output = 1 / (1 + np.exp(-input_data))
    assert np.allclose(output_data, expected_output), "Test failed!"

def test_backward():
    sigmoid_layer = SigmoidLayer()
    input_data = np.array([[-1, 0, 1], [2, -3, 4]])
    sigmoid_layer.Input = input_data
    delta = np.ones((2, 3))
    output_data = sigmoid_layer.backward(delta)
    expected_output = delta * input_data * (1 - input_data)
    print(output_data)
    print(expected_output)
    assert np.allclose(output_data, expected_output), "Test failed!"

test_sigmoid_layer()
print("Test passed!")
test_backward()
