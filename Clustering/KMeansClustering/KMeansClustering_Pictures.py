import numpy as np
import os
import random
import matplotlib.pyplot as plt, mpld3
from matplotlib import style
import pandas as pd 
import copy

''' 
    Clustering 136 pictures in 4 Clusters
    Author: Ivan Antunović
'''

class ClusteringResult:
    def __init__(self, centroids, clusterIndicatorMatrix, minError):
        self.clusterIndicatorMatrix = clusterIndicatorMatrix
        self.centroids = centroids
        self.minError = minError

    def getMinError(self):
        return self.minError

    def getCentroids(self):
        return self.centroids

    def getClusterIndicatorMatrix(self):
        return self.clusterIndicatorMatrix

def calculateMinClusteringResult(clusteringResults):

    minClusteringResult = None
    # find minimum error result
    minError = 0

    for clusteringResult in  clusteringResults :
        error = clusteringResult.getMinError()
        if error < minError:
            minError = error
            minClusteringResult = clusteringResult
    
    return clusteringResult

def orderPictures(pictures, clusteringResult):

    dict = {}
    for i in range(K):
	    dict[i] = []

    clusterIndicatorMatrix = clusteringResult.getClusterIndicatorMatrix()
    for i in range (0, pictures.shape[0] ):
   
       img = pictures[i].reshape((243,160) )
       if clusterIndicatorMatrix[i][0] == 1:
           dict[0].append(img)

       elif clusterIndicatorMatrix[i][1] == 1:
           dict[1].append(img)

       elif clusterIndicatorMatrix[i][2] == 1:
           dict[2].append(img)

       elif clusterIndicatorMatrix[i][3] == 1:
           dict[3].append(img)

    return dict

def loadImage(imgPath):
    listPath = os.listdir(imgPath)
    listPath.sort()
    X = []
    originalPictures = []
    count = 0
    for img in listPath:
        if img == "Readme.txt":
            continue
        else:
            im = plt.imread(imgPath+img) 
            # Every picture has is 3-D Matrix, dimensions: 1) height, width, color(RGB)
            # Flattern the data into one vector
            # Ignore last dimension (Color)
            im2 = im[:, :, 0]
            
            originalPictures.append(im2)
            X.append(im2.flatten())
            count += 1

    return np.asarray(X,dtype=np.float64 ), np.asarray(originalPictures,dtype=np.float64 ), count



# Initialize K clusters to the random data points 
def initClusterCentroids(X, K):

    centroids = X[ np.random.choice( np.arange(len(X) ), K) , :]
    return np.array(centroids)

def calculateClusteringError(X, mean, R, N, K):

    error = 0

    for n in range (0, N):
        for k in range (0, K):
            error += R[n][k] * np.linalg.norm( X[n] - mean[k] ) ** 2

    return error

def isConverged(centroids, centroidsOld):

    CONVERGED_DISTANCE = 1e-4
    
    if np.linalg.norm(centroids - centroidsOld) < CONVERGED_DISTANCE:
        return True

    return False

def kMeansClusteringAlgo(X, K, maxIterations = 20):

    centroids = initClusterCentroids(X, K)
    centroidsOld = np.zeros(centroids.shape) # To store the value of centroids when it updates

    # Initialize dictionary
    classes = {}
    for i in range(K):
        classes[i] = []

    errors = []
    iteration = 0
    clusterIndicatorMatrix = np.zeros( (X.shape[0], K) )

    while iteration < maxIterations:

        n = 0
        for dataPoint in X:

            distances =[]
            for k in range (0, K):
                distances.append( np.linalg.norm(dataPoint - centroids[k]) ** 2 )

            minIndex = distances.index( min(distances) )
            clusterIndicatorMatrix[n][minIndex] = 1
            classes[minIndex].append(dataPoint)
            
            n += 1
            
        for classification in classes:
            centroids[classification] = np.average(classes[classification], axis = 0 )

        errors.append(calculateClusteringError(X, centroids, clusterIndicatorMatrix, X.shape[0], K))   
        centroidsOld = np.copy(centroids)

        #if isConverged(centroids, centroidsOld):
        #    break
   
        iteration += 1
        print(iteration)

    return centroids, clusterIndicatorMatrix, errors.index( min(errors) )

def plotClusteredImages(pictures):

    for i in range (len(pictures)):

        plt.subplot(1, len(pictures), i + 1)
        plt.imshow(  pictures[i] ) 
        plt.xticks([])
        plt.yticks([])

    plt.show()


K = 4
X, originalPictures, cnt = loadImage("yalefaces_cropBackground/")

minErrors = []
clusteringResults = []

for i in range(10):

    print("Run #", i)
    centroids, clusterIndicatorMatrix, minError = kMeansClusteringAlgo(X, K, maxIterations=10)
    clusteringResults.append( ClusteringResult(centroids, clusterIndicatorMatrix, minError) )

dictionary = orderPictures(originalPictures, calculateMinClusteringResult(clusteringResults))

for i in range (K):

  plotClusteredImages( dictionary.get(i) )



