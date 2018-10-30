import os
import re
import numpy as np
import math

CURRENT_PATH = os.path.dirname(__file__)
trainSet = []
trainLable = []
testSet = []
testLabel = []


def getInputs():
    # options = str(input("Enter L to load trained network, T to train a new one, Q to quit: "))
    options = "t"  # test Purpose
    if options == "L" or options == "l":
        print("load")
    elif options == "T" or options == "t":
        print("train")
        while (True):
            # resolution = str(input("Resolution of data (5/10/15/20): "))
            resolution = "5"  # test Purpose
            if resolution != "5" and resolution != "10" and resolution != "15" and resolution != "20":
                continue
            else:
                loadData(resolution)
                return
    else:
        print("Goodbye.")
        return


def loadData(resolution):
    global trainSet, trainLable, testSet, testLabel
    if resolution == "5":
        resolution = "0" + resolution
    trainSetFileName = "trainSet_" + resolution + ".dat"
    testSetFileName = "testSet_" + resolution + ".dat"
    trainSetPath = CURRENT_PATH + "/" + "trainSet_data/" + trainSetFileName
    testSetFilePath = CURRENT_PATH + "/" + "testSet_data/" + testSetFileName
    if not (os.path.exists(trainSetPath) and os.path.exists(testSetFilePath)):
        print("Can not find the file, NN.py should in the same folder with the trainSet_data and testSet_data")
        return
    trainSet = []
    trainLable = []
    testSet = []
    testLabel = []
    with open(trainSetPath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line[0] == "#":
                continue
            regExpList = re.findall(r'[(](.*?)[)]', line)
            trainSet.append(list(map(float, regExpList[0].split(" "))))
            trainLable.append(list(map(float, regExpList[1].split(" "))))
    print("Load Training Set.")
    with open(testSetFilePath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line[0] == "#":
                continue
            regExpList = re.findall(r'[(](.*?)[)]', line)
            testSet.append(list(map(float, regExpList[0].split(" "))))
            testLabel.append(list(map(float, regExpList[1].split(" "))))
    print("Load Testing Set.")
    normaliseData(int(resolution))
    initNeurons(trainSetFileName, testSetFileName)


def normaliseData(resolution):
    global trainSet, testSet
    count = resolution * resolution
    for x in range(len(trainSet)):
        for i in range(len(trainSet[0]) - count, len(trainSet[0])):
            trainSet[x][i] = trainSet[x][i] / 255
    for x in range(len(testSet)):
        for i in range(len(testSet[0]) - count, len(testSet[0])):
            testSet[x][i] = testSet[x][i] / 255


def initNeurons(trainSetFileName, testSetFileName):
    print("Initializing network...")
    hiddenLayers = initHiddenLayer()
    inputLayer = initInputLayer(len(trainSet[0]))
    outputLayer = initOutputLayer(len(trainLable[0]))
    network = []
    network.append(inputLayer)
    network = network + hiddenLayers
    network.append(outputLayer)
    network = initWeight(network)
    trainedNetwork = backPropagationLearning(network, trainSetFileName)
    validateNetwork(trainedNetwork, testSetFileName)


def validateNetwork(network, testSetFileName):
    print("Testing on " + testSetFileName + "...")
    correct = 0
    for (image, lable) in zip(testSet, testLabel):
        countNeuron = 0
        for neu in network[0]:
            neu.setActivationFunctionOutput(image[countNeuron])
            countNeuron += 1
        for i in range(1, len(network)):
            for neo in network[i]:
                input = neo.weight
                a = getActivationOutputVector(network[i - 1])
                # print(len(a), "______________", neo.bias)
                newAVal = np.dot(a, input) + neo.bias  # can not reverse order (65, ) dot (65, 1)
                neo.setActivationFunctionOutput(sigmoid(newAVal))
        outputVector = []
        for neu in network[len(network) - 1]:
            outputVector.append(neu.aVal)
        maxVal = max(outputVector)           # always same (super close)value
        maxIndex = outputVector.index(maxVal)
        if lable[maxIndex] == 1.0:
            correct += 1
    print("Accuracy achieved: ", correct / len(testSet))


def backPropagationLearning(network, trainSetFileName):
    print("Training on " + trainSetFileName + "...")
    for epoch in range(10):                         # Change the epoche maxError did not change,and accurate
        maxError = 0.0
        # Propagate the inputs forward to compute the outputs
        for (timage, tlable) in zip(trainSet, trainLable):
            # Input layer
            # Activation output for input layer is input value
            countNeuron = 0
            for neu in network[0]:
                neu.setActivationFunctionOutput(timage[countNeuron])
                countNeuron += 1
            for i in range(1, len(network)):
                for neu in network[i]:
                    input = neu.weight
                    a = getActivationOutputVector(network[i - 1])
                    # print(len(a), "______________", neo.bias)
                    newAVal = np.dot(a, input) + neu.bias  # can not reverse order (65, ) dot (65, 1)
                    neu.setActivationFunctionOutput(sigmoid(newAVal))
            # Propagate deltas backward from output layer to input layer    Some problems
            # output layer
            countNeuron1 = 0
            deltaSet = []  # length: len(network) - 1  input layer do not have weights sets
            outputDeltaSet = []
            for neu in network[len(network) - 1]:
                y = tlable[countNeuron1]
                derivativeVal = getDerivativeVal(neu.aVal)
                if abs(y - neu.aVal) > maxError:
                    maxError = abs(y - neu.aVal)
                deltaVal = derivativeVal * (y - neu.aVal)
                outputDeltaSet.append(deltaVal)
                countNeuron1 += 1
            deltaSet.insert(0, outputDeltaSet)
            for i in range(len(network) - 2, 0, -1):
                hiddenDeltaSet = []
                count = 0
                for neu in network[i]:
                    # a = getActivationOutputVector(network[i - 1])
                    preWeight = getWeightVectorFromLastLayer(network, count, i + 1)
                    derivativeVal = getDerivativeVal(neu.aVal)
                    preDelta = deltaSet[0]
                    newDelta = getNewDeltaVal(preDelta, preWeight, derivativeVal)
                    hiddenDeltaSet.append(newDelta[0])
                    count += 1
                deltaSet.insert(0, hiddenDeltaSet)
            # Update every weight in network using deltas
            # Assume there is a correct delta dataset
            for i in range(1, len(network)):
                for n in range(len(network[i])):
                    neu = network[i][n]
                    newDelta = deltaSet[i - 1][n]
                    a = getActivationOutputVector(network[i - 1])
                    for j in range(len(neu.weight)):
                        neu.weight[j] = neu.weight[j] + newDelta * a[j]
                    neu.bias = neu.bias + newDelta
        if maxError < 0.01:
            print("Oppppps")
            return network
    return network


def getActivationOutputVector(layer):
    list = []
    for neu in layer:
        list.append(neu.aVal)
    list = np.array(list)
    return list


def sigmoid(z):
    s = 1 / (1 + math.exp(-z))
    # if s == 1:
    #     s = 0.999
    return s


def initWeightHelper(dim):
    w = np.random.rand(dim, 1)
    b = np.random.rand()
    return w, b


def initWeight(network):
    for i in range(1, len(network)):
        preLayerSize = len(network[i - 1])
        for neuron in network[i]:
            w, b = initWeightHelper(preLayerSize)
            neuron.setWeight(w)
            neuron.setBias(b)
    return network


def getNewDeltaVal(preDelta, oldWeight, derivativeVal):
    count = 0.0
    for (d,w) in zip(preDelta,oldWeight):
        count = count + d * w
    return count * derivativeVal


def getDerivativeVal(aVal):
    return aVal * (1 - aVal)


def getWeightVectorFromLastLayer(network, neuronIndex, layerIndex):
    vector = []
    layer = network[layerIndex]
    for neu in layer:
        vector.append(neu.weight[neuronIndex])
    return vector


def initHiddenLayer():
    hiddenLayerDepth = int(input("Number of hidden layers: "))
    hiddenLayers = []
    depthCount = 0
    while depthCount < hiddenLayerDepth:
        hiddenLayerSize = int(input("Size of hidden layer " + str(depthCount) + ":"))
        curLayer = []
        for i in range(hiddenLayerSize):
            tag = "hidden_" + str(depthCount) + "_neuron_" + str(i)
            neuron = Neuron(tag)
            curLayer.append(neuron)
        hiddenLayers.append(curLayer)
        depthCount += 1
    # print(hiddenLayers)
    return hiddenLayers


def initInputLayer(size):
    inputLayer = []
    for i in range(size):
        tag = "input_" + "neuron_" + str(i)
        inputNeuron = Neuron(tag)
        inputLayer.append(inputNeuron)
    return inputLayer


def initOutputLayer(size):
    outputLayer = []
    for i in range(size):
        tag = "output_" + "neuron_" + str(i)
        outputNeuron = Neuron(tag)
        outputLayer.append(outputNeuron)
    return outputLayer


class Neuron(object):
    def __init__(self, tag):
        self.tag = tag

    def setWeight(self, weight):
        self.weight = weight

    def setActivationFunctionOutput(self, a):
        self.aVal = a;

    def setBias(self, bias):
        self.bias = bias


if __name__ == "__main__":
    getInputs()
