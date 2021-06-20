import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

sns.set_style(style="whitegrid")


# load the data
df = pd.read_csv("./datasets/data1.csv", header=None, names=["X", "Y"])


# create X and Y arrays
xvar = np.array(df['X'], dtype=np.float64)
yvar = np.array(df['Y'], dtype=np.float64)

# Initialize weights
theta = np.zeros((2, 1))

# Axis Operations
xvar = xvar[np.newaxis, ...]
xvar = np.transpose(xvar)

# Add random values
rng = np.random.default_rng(12345)
xvar = xvar + rng.random((xvar.shape))
yvar = yvar[..., np.newaxis]
xvar = np.hstack((np.ones(xvar.shape), xvar))



def gradientDescent(xvar, yvar, theta, learning_rate = 0.01, epochs = 1, batch_size = 32):
    """
    A mini-batch gradient descent function to get optimal values of theta. 
    It can be further customized to with learning rate, batch size
    and epochs
    """

    i = 0
    iterations = (xvar.shape[0] // batch_size) + 1
    
    
    def hypothesis(x, theta):
        return np.dot(x, theta)
    

    def gradient(x, y, theta):
        h = hypothesis(x, theta)
        return np.dot(x.transpose(), (h - y))

    for _ in range(epochs):
        for i in range(iterations):
            xmini = xvar[i*batch_size:(i+1)*batch_size]
            ymini = yvar[i*batch_size:(i+1)*batch_size]
            theta = theta - learning_rate * gradient(xmini, ymini, theta)
        if xvar.shape[0] % batch_size != 0:
            xmini = xvar[i*batch_size:xvar.shape[0]]
            ymini = yvar[i*batch_size:xvar.shape[0]]
            theta = theta - learning_rate * gradient(xmini, ymini, theta)

    return theta
        


# return optimal theta values
theta = gradientDescent(xvar, yvar, theta, learning_rate = 0.001, epochs = 15)

# predict the output based on the given data
pred = np.dot(xvar, theta)

# Plot the output along with input data
plt.figure(figsize=(16, 6))
plt.scatter(xvar[:, 1], yvar[:, ], marker = '.')
plt.plot(xvar[:, 1], pred[:,], color = 'orange')
plt.show()
