# Made by Ivan Antunovic 

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
from mpl_toolkits.mplot3d.axes3d import Axes3D # 3D plotting

###############################################################################
# Helper functions
def mdot(*args):
    ret = args[0]
    for a in args[1:]:
        ret = dot(ret,a)
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

def getSquaredLoss(X_test, y_test, beta):
    error = y_test - X_test.dot(beta)
    squaredError = np.power( np.linalg.norm(error) , 2 )
    return squaredError

# calculates optimum for Ridge Regression
def calculateRidgeBeta(X, y, lamda = 0):
    
    size = np.size(X, 1)
    identityMatrix = np.identity(size)
    identityMatrix[0, 0] = 0 # with I[1,1] = 0 if BETA is not regularized
    
    beta = mdot( inv( dot(X.T, X) + lamda * identityMatrix ), X.T, y)
    return beta
        
def prependQuadraticFeatures(X):
    
    #prepend (1, x1, x1^2)
    if ( len(X.shape) == 1 ): # Length = 1, means we got column vector
        x1_squared = np.power( X, 2 )
        
        return np.column_stack( (np.ones(X.shape[0]), 
                                 X,
                                 x1_squared,) )
    
    #prepend (1, x1, x2, x1^2, x1 * x2, x2^2)
    elif( len(X.shape) == 2 ): 
        x1, x2 = X[:, 0], X[:, 1]
        x1_squared = np.power( x1, 2 )
        x2_squared = np.power( x2, 2 )
        
        return np.column_stack( (np.ones(X.shape[0]), 
                                 x1,
                                 x2,
                                 x1_squared,
                                 np.multiply(x1, x2),
                                 x2_squared) )
    
    return None
           
def getTrainingData(X, y, k, subsetIndex):

    numberOfRows = X.shape[0]
    offset = numberOfRows // k
    
    X_train_data = np.delete(X, np.s_[subsetIndex * offset : (subsetIndex + 1) * offset], axis=0) # remove rows, exclude subset: X[subsetIndex]
    y_train_data = np.delete(y, np.s_[subsetIndex * offset : (subsetIndex + 1) * offset], axis = 0) # remove rows, exclude subset: y[subsetIndex]
    
    return X_train_data, y_train_data
    
def getValidationData(X, y, k, subsetIndex):
      
    numberOfRows = X.shape[0]
    offset = numberOfRows // k
    
    try:
        X_validation_data = X[subsetIndex * offset : (subsetIndex + 1) * offset] #include X[subsetIndex]
        y_validation_data = y[subsetIndex * offset : (subsetIndex + 1) * offset] #include y[subsetIndex]
        
    except IndexError:
        print("Error while getting validation data")
        return None
    
    return X_validation_data, y_validation_data

def findBestLambda(lambdas, meanLestSquareLosses):
    
    min_loss_index = meanLestSquareLosses.index(min(meanLestSquareLosses))
    
    best_lambda = lambdas[min_loss_index]
    
    return best_lambda

def plotLossVsLambda(meanLestSquareLosses, lambdas, bestLambda):
    
    plt.figure()
    # Removed variance from plot to avoid fucking the scale up
    #plt.errorbar(loglambdas, cvloss[:, 0], yerr = cvloss[:, 1], fmt = '--')
    #plt.errorbar(loglambdas, cvloss[:, 0], fmt = '--')
    plt.xlabel('Lambda')
    plt.ylabel('Mean Squared Error')
    plt.axvline(bestLambda, color="red")
    plt.plot(lambdas, meanLestSquareLosses)


# Runs cross-validation with one specific value of lambda
def crossValidation(X, y, k, lambda_):

    sumLeastSquareLoss = 0
 
    for subsetIndex in range(k):
        
        # Divide the set into k subsets/folds
        temp_X = np.copy(X)
        temp_y = np.copy(y)
        
        X_train_data, y_train_data = getTrainingData(temp_X, temp_y, k, subsetIndex)
        X_validation_data, y_validation_data = getValidationData(temp_X, temp_y, k, subsetIndex)
        
        # 3. Compute BETA on training data
        beta = calculateRidgeBeta(X_train_data, y_train_data, lambda_)
        
        validation_data_num_rows = X_validation_data.shape[0]
        # 4. compute ERROR on validation data
        leastSquareLoss = getSquaredLoss(X_validation_data, y_validation_data, beta) / validation_data_num_rows # Divide with the number of rows of each data subset
                                                                             # To normalize the error
        sumLeastSquareLoss += leastSquareLoss
           
    # 6. Report mean squared error
    meanSquaredError = sumLeastSquareLoss / k
    
    return meanSquaredError
    
    
def multiCrossValidation(X, y, k = 10):
    
    lambdas = np.array([ ki for ki in range(0, 101, 1) ])
    
    X = prependQuadraticFeatures(X)
    y = prependQuadraticFeatures(y)
    
    meanLestSquareLosses = [None] * len(lambdas)
    
    print(len(lambdas))
    for lamda in range( len(lambdas) ):
        meanLestSquareLosses[lamda] = crossValidation(X, y, k = 10, lambda_= lamda)
        
    bestLambda = findBestLambda(lambdas, meanLestSquareLosses)
    print("Best Lambda: ", bestLambda)
    
    meanLestSquareLosses = np.asarray(meanLestSquareLosses)
    
    plotLossVsLambda(meanLestSquareLosses, lambdas, bestLambda)
    
    #betas = np.array([ ridgeRegression(Xfv, y, l) for l in lambdas ])
    #bestBeta = betas[bestLambda]
    
    #calculate f(x) = fi(x)^T * beta
    #function = paritioned_X.dot(bestBeta)
    
    #plot function
    #plot loss vs lambda
    
    
###############################################################################

# load the data
data = np.loadtxt("dataQuadReg2D_noisy.txt")

# split into features and labels
X, y = data[:, :2], data[:, 2]

multiCrossValidation(X, y, k = 10)


