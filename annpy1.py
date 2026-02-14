import numpy as np

x = np.array([100,17])

def sigmoid(z):
    x = 1 / (1 + np.exp(-z))
    return x

def _neural_network():
    w_1_1 = np.array([1,2])
    b_1_1 = np.array([-1])
    z_1_1 = np.dot(w_1_1,x) + b_1_1
    a_1_1 = sigmoid(z_1_1)

    w_1_2 = np.array([2,3])
    b_1_2 = np.array([-2])
    z_1_2 = np.dot(w_1_2,x) + b_1_2
    a_1_2 = sigmoid(z_1_2)

    w_1_3 = np.array([3,4])
    b_1_3 = np.array([-3])
    z_1_3 = np.dot(w_1_3,x) + b_1_3
    a_1_3 = sigmoid(z_1_3)

    a1 = np.array([a_1_1, a_1_2, a_1_3])

    w_2_1 = np.array([1,2,3])
    b_2_1 = np.array([-1])
    z_2_1 = np.dot(w_2_1,a1) + b_2_1
    a_2_1 = sigmoid(z_2_1)

    return a_2_1
    print(a_2_1)
_neural_network()