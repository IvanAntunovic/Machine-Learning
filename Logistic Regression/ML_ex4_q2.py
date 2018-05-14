# Author: Ivan Antunovic
#!/usr/bin/env python
# encoding: utf-8
"""
This is a mini demo of how to use numpy arrays and plot data.
NOTE: the operators + - * / are element wise operation. If you want
matrix multiplication use ‘‘dot‘‘ or ‘‘mdot‘‘!
"""
import numpy as np
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 3D plotting
import matplotlib.cm as cm


###############################################################################
# Helper functions
def mdot(*args):
    ret = args[0]
    for a in args[1:]:
        ret = dot(ret, a)
    return ret


def prepend_one(X):
    """prepend a one vector to X."""
    return np.column_stack([np.ones(X.shape[0]), X])


def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!
    """
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])


def calculateSigmoid(X, beta):
    functionArgument = dot(X, beta)
    return np.exp(functionArgument) / (1 + np.exp(functionArgument))  # size n x 1


def prependQuadraticFeatures(X):
    # prepend (1, x1, x2, x1^2, x1 * x2, x2^2)
    x1, x2 = X[:, 0], X[:, 1]
    x1_squared = np.power(x1, 2)
    x2_squared = np.power(x2, 2)

    return np.column_stack((np.ones(X.shape[0]),
                            x1,
                            x2,
                            x1_squared,
                            np.multiply(x1, x2),
                            x2_squared))


def prependLinearFeatures(X):
    # prepend (1, x1, x2)
    x1, x2 = X[:, 0], X[:, 1]

    return np.column_stack((np.ones(X.shape[0]),
                            x1,
                            x2))


def plotModel(X, y, beta, featureVector):
    X_grid = prependFeatureVector(grid2d(-3, 3, num=30), featureVector)

    y_grid = calculateSigmoid(X_grid, beta)

    labs = np.array(y == 1)

    # vis the result
    fig = plt.figure()
    ax = fig.add_subplot(211, projection='3d')  # the projection part is important
    ax.scatter(X_grid[:, 1], X_grid[:, 2], y_grid)  # don't use the 1 infront
    ax.scatter(X[:, 1], X[:, 2], y)  # also show the real data
    ax.set_title("predicted data")
    
    return plt

def plotData(data):
    X = data[:, :2]
    x1 = []
    x2 = []

    for point in data:
        if point[2] == 1:
            x1.append(point[0])
            x2.append(point[1])

    plt.plot(x1, x2, 'ro')

    x1 = []
    x2 = []

    for point in data:
        if point[2] == 0:
            x1.append(point[0])
            x2.append(point[1])

    plt.plot(x1, x2, 'go')

    return plt

def prependFeatureVector(X, featureVector):

    if featureVector == "linear":
        return prependLinearFeatures(X)

    elif featureVector == "quadratic":
        return prependQuadraticFeatures(X)

##############################################################################
class LogisticRegression:

    # Calculates gradient = X.T * (p - y) + 2*lamda*I*beta
    def calculateGradient(self, X, y, beta, lamda):
        p = calculateSigmoid(X, beta)

        # Calculate gradient dimension n x 1
        return mdot(X.transpose(), (p - y)) + (2 * lamda) * mdot(np.identity(X.shape[1]), beta)

    # Calculates Hessian = X.T*W*X + 2*lambda*I
    # , where W = diag(p  (1 - p)),  = element-wise product of two matricies
    def calculateHessian(self, X, beta, lamda):
        p = calculateSigmoid(X, beta)

        # element-wise product of two matricies
        W = np.diag(np.multiply(p, 1 - p))

        # Calculate Hessian of dimension n x n
        return mdot(X.transpose(), W, X) + 2 * lamda * np.identity(X.shape[1])

    # Calculate optima beta using Newton-Raphson method
    # beta_new = beta_old - hessian^-1 * gradient
    def newtonMethod(self, X, y, lamda, stepCount=50):
        k = X.shape[1]
        # Initialize Beta to zero matrix
        beta = np.zeros(k)

        for i in range(stepCount):
            gradient = self.calculateGradient(X, y, beta, lamda)
            hessian = self.calculateHessian(X, beta, lamda)

            beta = beta - mdot(inv(hessian), gradient)

        return beta  # dimension n x 1

    def predict(self, X, y, stepCount=10):
        lamda = 0.01

        beta = self.newtonMethod(X, y, lamda, stepCount)

        return beta


###############################################################################

# load the data
data = np.loadtxt("data2Class.txt")

# split into features and labels
X, y = data[:, :2], data[:, 2]

logisticRegression = LogisticRegression()

featureVector = "linear"
X = prependFeatureVector(X, featureVector)
print(X.shape)
beta = logisticRegression.predict(X, y, 10)
print ("Calculated beta using Logistic Regression with" , featureVector, "features: \n", beta)

plotData(data)
plt = plotModel(X, y, beta, featureVector)
plt.show()


