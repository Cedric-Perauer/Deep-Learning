import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

data = pd.read_csv('data.csv')
y = data.iloc[:,2]
y = y.to_numpy()
x = data.iloc[:,:-1]
x = x.to_numpy()
x_new = np.zeros((x.shape[0]+1,2))
x_new[0][:] = [0.78051,-0.063669]
x_new[1:][:] = x
y_new = np.zeros(y.shape[0]+1)
y_new[0] = 1
y_new[1:] = y
params = trainPerceptronAlgorithm(x_new,y_new,0.01,100)


colors = [i for i in y_new]
for i in range(len(params)): 
    plt.clf()
    plt.scatter(x_new[:,0],x_new[:,1],c=colors)
    slope, intercept = params[i][0].item(),params[i][1].item()
    ys = x_new *slope + intercept
    plt.plot(x_new,ys)
    plt.draw()
    plt.pause(0.1)
plt.show()
