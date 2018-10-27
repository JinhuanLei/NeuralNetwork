import os
import re
import numpy as np

CURRENT_PATH = os.path.dirname(__file__)


def getInputs():
    options = str(input("Enter L to load trained network, T to train a new one, Q to quit: "))
    if options == "L" or options == "l":
        print("load")
    elif options == "T" or options == "t":
        print("train")
        while (True):
            resolution = str(input("Resolution of data (5/10/15/20): "))
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
    initNeurons(trainSet, trainLable, testSet, testLabel)


def initWeightHelper(dim):
    w = np.random.rand(dim, 1)
    b = np.random.rand()
    return w, b


def initWeight(network):
    for i in range(1,len(network)):
        preLayerSize = len(network[i-1])
        for neuron in network[i]:
            w,b = initWeightHelper(preLayerSize)
            neuron.setWeight(w)
            neuron.setBias(b)
    return network





def initNeurons(trainSet, trainLable, testSet, testLabel):
    hiddenLayers = initHiddenLayer()
    inputLayer = initInputLayer(len(trainSet[0]))
    outputLayer = initOutputLayer(len(trainLable[0]))
    network = []
    network.append(inputLayer)
    network = network + hiddenLayers
    network.append(outputLayer)
    network =initWeight(network)
    print(network[1][0].bias)





def initHiddenLayer():
    hiddenLayerDepth = int(input("Number of hidden layers: "))
    hiddenLayers = []
    depthCount = 0
    while depthCount < hiddenLayerDepth:
        hiddenLayerSize = int(input("Size of hidden layer " + str(depthCount) + ":"))
        curLayer = []
        for i in range(hiddenLayerSize):
            tag = "hidden_" + str(depthCount) + "_neuron_" + str(i)
            neuron = Neuron(tag, [])
            curLayer.append(neuron)
        hiddenLayers.append(curLayer)
        depthCount += 1
    # print(hiddenLayers)
    return hiddenLayers


def initInputLayer(size):
    inputLayer = []
    for i in range(size):
        tag = "input_" + "neuron_" + str(i)
        inputNeuron = Neuron(tag, [])
        inputLayer.append(inputNeuron)
    return inputLayer


def initOutputLayer(size):
    outputLayer = []
    for i in range(size):
        tag = "output_" + "neuron_" + str(i)
        outputNeuron = Neuron(tag, [])
        outputLayer.append(outputNeuron)
    return outputLayer


class Neuron(object):
    def __init__(self, tag, linkedNeurons):
        self.tag = tag
        self.linkedNeurons = linkedNeurons

    def setWeight(self, weight):
        self.weight = weight


    def setBias(self, bias):
        self.bias = bias




if __name__ == "__main__":
    getInputs()
