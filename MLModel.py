import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1,2,3,4,5,6])
y_train = np.array([2,3,4,5,6,7])

def cost(x,y,w,b):
    m = x.shape[0]
    total_error = 0
    for i in range(m):
        f = w * x[i] + b
        error = f - y[i]
        total_error += error ** 2
    cost = (1 / (2 * m)) * total_error
    return cost

def gradient_descent(x,y,w,b,alpha):
    m = x.shape[0]

    for _ in range(1000):
        dj_dw = 0
        dj_db = 0

        for i in range(m):
            error = w * x[i] + b - y[i]
            dj_dw += error * x[i]
            dj_db += error

        dj_dw = (1/m)*dj_dw
        dj_db = (1/m)*dj_db

        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w,b

def predicts(x,y,w,b):
    m = x.shape[0]
    y_predicted = np.zeros(m)
    for i in range(m):
        f = w * x[i] + b
        y_predicted[i] = f
    return y_predicted
    
##################################################################

# w,b = gradient_descent(x_train, y_train, 100, 100, 0.1)
# predicted_outputs = predicts(x_train,y_train,w,b)

# plt.plot(x_train,predicted_outputs , c='b') # plotting the prediction line
# plt.scatter(x_train, y_train, marker='x', c='r')

###################################################################

x_train_2 = np.array([2.4,1.8,3,4])
y_train_2 = np.array([2.8,2.7,3,3.2])

w_2,b_2 = gradient_descent(x_train_2,y_train_2,100,100,0.1)
predicted_outputs_2 = predicts(x_train_2,y_train_2,w_2,b_2)
print(predicted_outputs_2)

plt.plot(x_train_2,predicted_outputs_2 , c='b') # plotting the prediction line
plt.scatter(x_train_2, y_train_2, marker='x', c='r')

