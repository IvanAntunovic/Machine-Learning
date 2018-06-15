# Author: Ivan Antunovic
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import scipy as sp
import os

# load all images into a big data matrix X 165 x 77760
def loadImage(imgPath): 

    pathsList = os.listdir(imgPath)
    pathsList.sort()
    X = []
    count = 0
    
    for fileName in pathsList:
        if fileName == "Readme.txt":
            continue
        else:
            # Read an image from a file into an array.
            im = plt.imread(imgPath + fileName)

            X.append(im.flatten())
            count += 1

    return np.asarray(X, dtype=np.float64 ), count


# QUESTION FOR THIS METHOD for centring the data, 
# why do we center the data
def meanAndCenter(Data):
    X_centered = np.copy(Data)
    mean = np.mean(Data, axis=0) # Calculate mean column-wise

    columns, rows = Data.shape

    return mean, X_centered - np.dot( np.ones(rows), mean );


def svd(Data):
    # u, s, vt = sla.svds(Data, k=imgNumber)
    u, s, vt = np.linalg.svd(Data, full_matrices=False)
    return u, s, vt

def reconstruct(X,mean,imgNumber,Z,Vp):

    # Reconstruct all faces by computing
    X_new = np.dot(Z,Vp.T)

    error = 0

    for i in range(0, imgNumber):
        X_new[i] += mean
        error += np.linalg.norm(X[i] - X_new[i])**2

    return X_new, error

def reconstruct(X_tilda, V, mean, p):

    Vp = V[:,0:p] 
    Z = np.dot(X_tilda, Vp)
    
    number_columns = mean.shape
    X_prime = np.dot( np.ones(number_columns), mean ) + np.dot( Z, Vp.T ) 

    return X_prime

def calculateError(X, X_prime):

    if len(X) != len(X_prime):
        print(" Error occured while calculating error")
        return

    errorSum = 0
    for i in range ( 0, len(X) ):
        errorSum += la.norm( ( X[:, i] - X_prime[:, i] ) )  ** 2

    return errorSum

def plotError(error, p):

    plt.figure()

    plt.xlabel('p')
    plt.ylabel('Error')
    plt.plot(p, error)
    plt.show()

def plotReconstructedImages(X_prime, numIterations):

    if numIterations > len(X_prime):
        print("Param Error while plotting Reconstructed Images")
        return

    for i in range (0, numIterations):
        chooseImage = X_prime[i]
        chooseImage = chooseImage.reshape((243,320))
        imgplot = plt.imshow(chooseImage, cmap='gray')
        plt.show()

imgPath = "yalefaces/"
X, imgNumber = loadImage(imgPath) #part a
mean, X_centered = meanAndCenter(X) #part b
u,s,vt = svd(X_centered) #part c
V = vt.T #get V

errorList = []
pList = []

for p in range(0 , 100, 10):
    X_prime = reconstruct(X_tilda = X, V = V, mean = mean, p = 100)
    error = calculateError(X = X, X_prime = X_prime)

    errorList.append( error )
    pList.append( p )

plotError(error = errorList, p = pList  )
plotReconstructedImages(X_prime, 2)
plotReconstructedImages(X_prime, 2)
   
