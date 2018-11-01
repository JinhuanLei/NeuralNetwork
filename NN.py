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
    global trainSet,trainLable
    print("Initializing network...")
    hiddenLayers = initHiddenLayer()
    inputLayer = initInputLayer(len(trainSet[0]))
    outputLayer = initOutputLayer(len(trainLable[0]))
    network = []
    network.append(inputLayer)
    network = network + hiddenLayers
    network.append(outputLayer)
    len(testSet)
    initWeight(network)
    #####################
    # trainSet = np.array(trainSet)
    # trainLable = np.array(trainLable)
    print("Training on " + trainSetFileName + "...")
    backPropagationLearning(network, 99)
    print("Testing on " + testSetFileName + "...")
    validateNetwork(network)



def backPropagationLearning(network, epoches):
    for epoch in range(epoches):
        maxError = 0
        for (image, label) in zip(trainSet, trainLable):
            # input layer
            for i in range(len(network[0])):
                network[0][i].setActivationFunctionOutput(image[i])
            # Next Layer
            for i in range(1, len(network)):
                preResult = getActivationOutputVector(network[i - 1])
                for j in range(len(network[i])):
                    weight = network[i][j].weight
                    bias = network[i][j].bias
                    newResult = np.dot(preResult, weight) + bias
                    network[i][j].setActivationFunctionOutput(sigmoid(newResult))
            # Back Ward
            # Output Layer delta
            layerDepth = len(network) - 1
            for i in range(len(network[layerDepth])):
                neu = network[layerDepth][i]
                error = label[i] - neu.aVal
                predict = 0
                # if neu.aVal>0.5:
                #     predict = 1
                # else:
                #     predict = 0
                # error = math.log(neu.aVal) * predict + (1-predict)* math.log(1-neu.aVal)
                if abs(error) > maxError:
                    maxError = abs(error)
                deriVal = getDerivativeVal(neu.aVal)
                delta = error * deriVal
                neu.setDelta(delta)
            # others layer delta
            for i in range(layerDepth - 1, 0, -1):
                for j in range(len(network[i])):
                    neu = network[i][j]
                    deriVal = getDerivativeVal(neu.aVal)
                    weight = getWeightVectorFromLastLayer(network[i + 1], j)
                    deltas = getDeltaVector(network[i + 1])
                    neu.delta = getNewDeltaVal(deltas, weight, deriVal)
            for i in range(1, len(network)):
                a = getActivationOutputVector(network[i - 1])
                for j in range(len(network[i])):
                    neu = network[i][j]
                    for z in range(len(neu.weight)):
                        neu.weight[z] = neu.weight[z] + a[z] * neu.delta
                        neu.bias = neu.bias + neu.delta
        if maxError < 0.01:
            print("Oppppps")
            return network
        print("Epoch ", epoch, ":")
        validateNetwork(network)
    return network


def validateNetwork(network):
    # global testSet, testLabel
    # testSet =trainSet
    # testLabel =trainLable
    # testLabel = testLabel[1:2]
    # testSet = []
    # ss = []
    # for i in range(65):
    #     ss.append(999)
    # testSet.append(ss)
    # testSet = trainSet[0:1]
    # testLabel = trainLable[0:1]
    correct = 0
    # print(len(testSet), len(testLabel))
    for (image, lable) in zip(testSet, testLabel):
        for i in range(len(network[0])):
            network[0][i].setActivationFunctionOutput(image[i])
        # Next Layer
        for i in range(1, len(network)):
            preResult = getActivationOutputVector(network[i - 1])
            for j in range(len(network[i])):
                weight = network[i][j].weight
                bias = network[i][j].bias
                newResult = np.dot(preResult, weight) + bias
                network[i][j].setActivationFunctionOutput(sigmoid(newResult))
        outputVector = []
        for neu in network[len(network) - 1]:
            outputVector.append(neu.aVal)
        maxVal = max(outputVector)  # always same (super close)value
        maxIndex = outputVector.index(maxVal)
        if lable[maxIndex] == 1.0:
            correct += 1
    print("Accuracy achieved: ", correct / len(testSet))


def getDeltaVector(layer):
    vector = []
    for neu in layer:
        vector.append(neu.delta)
    return vector


def getActivationOutputVector(layer):
    list = []
    for neu in layer:
        list.append(neu.aVal)
    list = np.array(list)
    return list


def sigmoid(z):
    # print(z)
    if z < -50:
        z = -50
    s = 1.0 / (1.0 + math.exp(-z))
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



def getNewDeltaVal(preDelta, oldWeight, derivativeVal):
    count = 0.0
    for (d, w) in zip(preDelta, oldWeight):
        count = count + d * w
    return count * derivativeVal


def getDerivativeVal(aVal):  # if get value too small set 0.001
    deri = aVal * (1 - aVal)
    # if deri < 0.01:
    #     deri = 0.01
    return deri


def getWeightVectorFromLastLayer(layer, neuronIndex):
    vector = []
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

    def setDelta(self, delta):
        self.delta = delta

    def setActivationFunctionOutput(self, a):
        self.aVal = a;

    def setBias(self, bias):
        self.bias = bias


if __name__ == "__main__":
    getInputs()
