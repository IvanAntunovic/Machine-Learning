import numpy as np
import matplotlib.pyplot as plt
from random import randint

def plotData(data):

    x = data[:,0]
    y = data[:,1]
   
    plt.scatter(x, y)
    plt.show()


def initCentroids(X, K):

    centroids = np.empty( (K, 2) )
    n, d = X.shape

    #Initialize by choosing the three means k to be different randomly selected data points xi
    for i in range(0, K):
        randomDataPoint = randint(0, n)
        centroids[i][0] = X[randomDataPoint, 0]
        centroids[i][1] = X[randomDataPoint, 1]

    return centroids

# Returns vector of covariance matrices
def initCovarianceMatrix(K):

    covarianceMatrixList = []
    for i in range(0, K):
        # Append Sigma/Covariance Matrix inizialized to Identity Matrix
        covarianceMatrixList.append( np.eye(2) )

    return np.asarray(covarianceMatrixList )

def initPiKs(K):

    pisList = []

    # Initialize pis to uniform distribution
    for k in range (0, K):
        pisList.append( 1 / K )

    return np.asarray(pisList)

# Calculates PDF
def calculatePDF( x_i, mean_k, covarianceMatrix_k):

    #constant = ( (2 * np.pi) ** (-n / 2.0) ) * ( np.linalg.det(covarianceMatrix_k) ** -0.5 )  
    constant = np.linalg.det( 2 * np.pi * covarianceMatrix_k ) ** -0.5 
    exponent = -0.5 * np.dot( np.dot( (x_i - mean_k).T, np.linalg.inv(covarianceMatrix_k)), (x_i - mean_k) )
    
    return constant * np.exp(exponent)


# Evaluate the posterior probability yik that point xi belongs to cluster k:
def calculatePosteriorProbabilitiy (k, x_i, pis, covarianceMatrix, mean):

    denominator = 0
    for j in range(0, K):

        denominator += pis[j] * calculatePDF( x_i, mean[j], covarianceMatrix[j] )

    nominator = pis[k] * calculatePDF( x_i, mean[k], covarianceMatrix[k])

    return nominator / denominator


def calculateNks(N, K, X, covarianceMatrix, mean, pisArray):

    N_arr = np.zeros(K)
    for k in range (0, K):

        for i in range(0, N):
            N_arr[k] += calculatePosteriorProbabilitiy(k, X[i], pisArray, covarianceMatrix, mean)

    return N_arr

def updateEstimates(N, X, meanArray, covarianceMatrixArray, pisArray):

    N_arr = calculateNks(N, K, X, covarianceMatrix, meanArray, pisArray )

    for k in range (0, K):

        # Update the estimates
        pisArray[k] = N_arr[k] / N
    
        for i in range(0, N):

            meanArray[k] +=  calculatePosteriorProbabilitiy(k, X[i], pisArray, covarianceMatrix, meanArray) * X[i]
        
            vectorDifference = X[i] - meanArray[k]
            covarianceMatrixArray[k] += calculatePosteriorProbabilitiy(k, X[i], pisArray, covarianceMatrix, meanArray) * np.dot(vectorDifference, vectorDifference.T) 

        meanArray[k] = meanArray[k] / N_arr[k] 
        covarianceMatrixArray[k] = covarianceMatrixArray[k] / N_arr[k] 


def gaussianMixtureModel(K, X, meanArray, pisArray, covarianceMatrixArray, ):

    # Number of data points
    N = X.shape[0]

    isConverged = False
    CONVERGENCE_VALUE = 1e-4
    iterations = 0;
    maxIterations = 1000

    while not isConverged or iterations < maxIterations:

        updateEstimates(N, X, meanArray, covarianceMatrixArray, pisArray)

        # Evaluate log likelihood & Check for convergence
        logLikelihood = calculateLogLikelihood(X, N, K, pisArray, covarianceMatrixArray, meanArray)
        if logLikelihood < CONVERGENCE_VALUE:
            isConverged = True
        print ("Log likelihood: ", logLikelihood)
        
        iterations += 1

def calculateLogLikelihood(X, N, K, pisArray, covarianceMatrixArray, meanArray ):

    logLikelihood = 0
    for n in range (0, N):
        
        logLikelihoodArgument = 0
        for k in range (0, K):
           
            logLikelihoodArgument += pisArray[k] * calculatePDF( X[n], meanArray[k], covarianceMatrix[k] )

        logLikelihood += np.log( logLikelihoodArgument )

    return logLikelihood

# Load data
X = np.loadtxt("mixture.txt")
plotData(X)

K = 3
# Initialize
centroids = initCentroids(X, K)
covarianceMatrix = initCovarianceMatrix(K)
pisArray = initPiKs(K)

print("Calling Gaussian Mixture Model ...")
gaussianMixtureModel(K, X, centroids, pisArray, covarianceMatrix)
