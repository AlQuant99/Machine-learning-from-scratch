import sklearn.datasets
import matplotlib.pyplot as plt
import numpy as np
X, Y = sklearn.datasets.make_moons(n_samples=500, noise=.2)
X, Y = X.T, Y.reshape(1, Y.shape[0])
epochs = 1000
learningrate = 0.01
def sigmoid(z):
  return 1 / (1 + np.exp(-z))
losstrack = []
m = X.shape[1]
w = np.random.randn(X.shape[0], 1)*0.01
b = 0
for epoch in range(epochs):
  z = np.dot(w.T, X) + b
  p = sigmoid(z)
  cost = -np.sum(np.multiply(np.log(p), Y) + np.multiply((1 - Y), np.log(1 - p)))/m
  losstrack.append(np.squeeze(cost))
  dz = p-Y
  dw = (1 / m) * np.dot(X, dz.T)
  db = (1 / m) * np.sum(dz)
  w = w - learningrate * dw 
  b = b - learningrate * db
plt.plot(losstrack)
