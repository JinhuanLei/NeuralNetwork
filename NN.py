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
                print("Load Training Set.")
                print("Load Testing Set.")
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
    with open(testSetFilePath, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if line[0] == "#":
                continue
            regExpList = re.findall(r'[(](.*?)[)]', line)
            testSet.append(list(map(float, regExpList[0].split(" "))))
            testLabel.append(list(map(float, regExpList[1].split(" "))))
    initNeurons(trainSetFileName, testSetFileName)


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
        countNeuron = 0
        flag = 1
        for neu in network[len(network) - 1]:
            y = lable[countNeuron]
            judge = 0
            if(neu.aVal>0):
                judge = 1
            else:
                judge = 0
            if y != judge:
                flag = 0
                break
            countNeuron += 1
        if flag == 1:
            correct += 1
    print("Accuracy achieved: ", correct / len(testSet))


def backPropagationLearning(network, trainSetFileName):
    # Propagate the inputs forward to compute the outputs
    print("Training on " + trainSetFileName + "...")
    for num in range(50):
        for (timage, tlable) in zip(trainSet, trainLable):
            # input layer
            countNeuron = 0
            for neu in network[0]:
                neu.setActivationFunctionOutput(timage[countNeuron])
                countNeuron += 1
            for i in range(1, len(network)):
                for neo in network[i]:
                    input = neo.weight
                    a = getActivationOutputVector(network[i - 1])
                    # print(len(a), "______________", neo.bias)
                    newAVal = np.dot(a, input) + neo.bias  # can not reverse order (65, ) dot (65, 1)
                    neo.setActivationFunctionOutput(sigmoid(newAVal))
            # Propagate deltas backward from output layer to input layer    Some problems
            # output layer
            countNeuron = 0
            deltaVal = 0
            for neu in network[len(network) - 1]:
                y = tlable[countNeuron]
                derivativeVal = getDerivativeVal(neu.aVal)
                # if y - neu.aVal < 0.01:
                #     print("Opps")
                #     return network
                deltaVal = derivativeVal * (y - neu.aVal)
                countNeuron += 1
                # Update every weight in network using deltas
            for i in range(len(network) - 2, 0, -1):
                count = 0
                for neu in network[i]:
                    a = getActivationOutputVector(network[i - 1])
                    oldWeight = getWeightVectorFromLastLayer(network, count, i + 1)
                    derivativeVal = getDerivativeVal(neu.aVal)
                    newDelta = getNewDeltaVal(deltaVal, oldWeight, derivativeVal)
                    for j in range(len(neu.weight)):
                        neu.weight[j] = neu.weight[j] + newDelta * a[j]
                    neu.bias = neu.bias + newDelta  # dummy value always 1 ??
                    count += 1
    return network


def getActivationOutputVector(layer):
    list = []
    for neu in layer:
        list.append(neu.aVal)
    list = np.array(list)
    return list


def sigmoid(z):
    s = 1 / (1 + math.exp(-z))
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


def getNewDeltaVal(deltaVal, oldWeight, derivativeVal):
    count = 0.0
    for w in oldWeight:
        count = count + deltaVal * w
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
