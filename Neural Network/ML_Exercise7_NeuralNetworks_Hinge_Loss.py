
################################################################################################################################
#   Neural Network classifier implementation for 2 classes, using 1 Hidden layer                                               #                                                                                                           
#   Loss function: Hinge loss                                                                                              # 
################################################################################################################################
import numpy as np
from scipy.stats import logistic
import matplotlib.pyplot as plt
from matplotlib import style

def loadData(fileName="data2Class_adjusted.txt"):

    data = np.loadtxt(fileName)
    y = data[ :, 3 ]
    X = np.delete( data, data.shape[1] - 1, axis = 1 )

    return X, y

def initializeLayers(numberInputs = 3, numberOutputs = 1, numUnitsInHiddenLayer = 100):
    
    weights = []
    weights.append( np.random.uniform(low=-1.0, high=1.0, size=(numUnitsInHiddenLayer, numberInputs) ) )
    weights.append( np.random.uniform(low=-1.0, high=1.0, size=(numberOutputs, numUnitsInHiddenLayer) ) )

    return np.asarray(weights)

def calculateSigmoid(param):
    
    return np.exp(param) / ( 1 + np.exp(param) )  
  
# Calculates Hinge loss
def calcLoss(hypothesis, labeled_output):
    
    return np.maximum(0, 1 - labeled_output * hypothesis)

def calcDerivativeLossWrtHypothesis(hypothesis, labeled_output):
    
    if 1 - labeled_output * hypothesis > 0:
        return np.array(-labeled_output)

    return np.array(0)


# Runs forward propagation and calculate hypothesis (predicted output) of the classifier
def forwardPropagation(x, weights):

    # Calculate activation in layer 1
    # gives 100 x 1 column vector
    activation_layer_1 = logistic.cdf( np.dot(weights[0], x) )

    # Calculate hypothesis (scalar value)
    hypothesis = np.dot(weights[1],  activation_layer_1) 
    # Map hypothesis to the probability by applying sigmoid
    #hypothesis = calculateSigmoid(hypothesis[0])

    return hypothesis, np.asarray( activation_layer_1 )

def backwardPropagation(hypothesis, x, labeled_output, activation_layer):

    activation_layer = np.resize( activation_layer, (activation_layer.shape[0], 1) )
    x = np.resize( x, (x.shape[0], 1) )

    # Produces scalar value
    delta_2 = calcDerivativeLossWrtHypothesis(hypothesis, labeled_output)
    delta_1 = np.multiply(  delta_2 * weights[1] ,  np.multiply(activation_layer, 1 - activation_layer).T )

    lossWrtWeights_1 =  delta_2.T * activation_layer.T 
    lossWrtWeights_0 = np.dot( delta_1.T, x.T )

    return lossWrtWeights_0, lossWrtWeights_1 

# Calculates gradient descent and updates the weights
def updateWeights(weights, layerGradients, updateStepSize = 0.05):

    CONVERGENCE_VAL = 1e-3
    old_weights = np.copy(weights)
    
    weights[0] = old_weights[0] - updateStepSize * layerGradients[0]
    weights[1] = old_weights[1] - updateStepSize * layerGradients[1]

    if np.abs( np.linalg.norm( weights[0] ) - np.linalg.norm( old_weights[0] )) < CONVERGENCE_VAL and np.abs( np.linalg.norm( weights[1] ) - np.linalg.norm( old_weights[1] ) < CONVERGENCE_VAL):
        return True
    
    return False


def trainNeuralNetwork(weights, X, labeledOutputs, numIterations = 10000):

    layerGradients = np.array( [np.zeros( (100, 3) ), np.zeros( (1, 100) )] ) 

    for iteration in range(numIterations):

        lossSum = 0
        for labelIndex in range (0, len(X)):
        
            hypothesis, activation_layer = forwardPropagation(X[labelIndex], weights)

            # Calculate neg-log likelikehood
            lossSum += calcLoss( hypothesis, labeledOutputs[labelIndex] )
            
            # Calculate gradients
            layerGradient_0, layerGradient_1 = backwardPropagation( hypothesis, X[labelIndex], labeledOutputs[labelIndex], activation_layer ) 
            layerGradients[0] += layerGradient_0
            layerGradients[1] += layerGradient_1
    
        print(iteration, ". ", lossSum)

        # Update the weights and check if gradient descent converges
        if updateWeights(weights, layerGradients):
            break

def predict(X, labeledOutputs, weights):

    lossSum = 0
    predictions = np.zeros( (200, 2) )

    for labelIndex in range (0, len(X)):
        
        hypothesis, activation_layer = forwardPropagation(X[labelIndex], weights)
        predictions[labelIndex] = np.array( [labeledOutputs[labelIndex], hypothesis] )

        # Calculate neg-log likelikehood
        lossSum += calcLoss( hypothesis, labeledOutputs[labelIndex] )

    return lossSum, predictions
            
X, y = loadData()

weights = initializeLayers()
trainNeuralNetwork(weights, X, y)
loss, predictions = predict(X, y, weights)

print(predictions)




    