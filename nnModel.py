import os
import re
import numpy as np
import random
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
    initNN(trainSetFileName, testSetFileName)


def normaliseData(resolution):
    global trainSet, testSet
    count = resolution * resolution
    for x in range(len(trainSet)):
        for i in range(len(trainSet[0]) - count, len(trainSet[0])):
            trainSet[x][i] = trainSet[x][i] / 255
    for x in range(len(testSet)):
        for i in range(len(testSet[0]) - count, len(testSet[0])):
            testSet[x][i] = testSet[x][i] / 255


def initNN(trainSetFileName, testSetFileName):
    nnPram = []
    hiddenLayers = initHiddenLayer()
    nnPram.append(len(trainSet[0]))
    nnPram = nnPram + hiddenLayers
    nnPram.append(len(trainLable[0]))
    trainData = list(zip(trainSet, trainLable))
    testData = list(zip(testSet, testLabel))
    nn = Network(nnPram)
    nn.train(trainData, testData, 1000, 10)


def initHiddenLayer():
    hiddenLayerDepth = int(input("Number of hidden layers: "))
    hiddenLayers = []
    depthCount = 0
    while depthCount < hiddenLayerDepth:
        hiddenLayerSize = int(input("Size of hidden layer " + str(depthCount) + ":"))
        hiddenLayers.append(hiddenLayerSize)
        depthCount += 1
    return hiddenLayers


class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = self.initBias()
        self.weights = self.initWeights()

    def initWeights(self):
        list = []
        for i in range(1,len(self.sizes)):
            list.append(np.random.randn(self.sizes[i] , self.sizes[i-1]))
        return list

    def initBias(self):
        list = []
        for i in range(1,len(self.sizes)):
            list.append(np.random.randn(self.sizes[i] , 1))
        return list

    def feedForward(self, a):
        for (b, w) in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, test_data, epochs ,batch):
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch] for k in range(0, len(training_data))]
            for mini_batch in mini_batches:
                self.updateWeight(mini_batch)
            print("epoch :", i)
            print("training Accurate :", self.evaluate(training_data))
            print("testing Accurate :", self.evaluate(test_data))

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def updateWeight(self,batchs):
        batch_bias = [np.zeros(b.shape) for b in self.biases]
        batch_weight = [np.zeros(w.shape) for w in self.weights]
        for x, y in batchs:
            x = self.matrixTranspose(x)
            y = self.matrixTranspose(y)
            delta_bias, delta_weight = self.backprop(x, y)
            batch_bias = [b + delta for b, delta in zip(batch_bias, delta_bias)]
            batch_weight = [weight + delta for weight, delta in zip(batch_weight, delta_weight)]
        self.weights = [w - (1 / len(batchs)) * nw
                        for w, nw in zip(self.weights, batch_weight)]
        self.biases = [b - (1 / len(batchs)) * nb
                       for b, nb in zip(self.biases, batch_bias)]

    def evaluate(self, test_data):
        test_results = [(self.feedForward(self.matrixTranspose(x)), self.matrixTranspose(y)) for (x, y) in test_data]
        count = 0
        for (x, y) in test_results:
            maxVal = x.max()
            maxIndex = (np.where(x == maxVal))[0]
            # print(y[maxIndex], maxVal)
            if y[maxIndex] == 1.0:
                count += 1
        return count / len(test_results)

    def matrixTranspose(self, x):
        for i in range(len(x)):
            if np.isnan(x[i]):
                x[i] = 0
        x = np.array(x)
        x = x.reshape(-1, 1)
        return x

    def backprop(self, x, y):
        biasMatrix = [np.zeros(b.shape) for b in self.biases]
        weightMatrix = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        outputMatrix = []
        for (b, w) in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            outputMatrix.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.getError(activations[-1], y) * self.getDerivativeVal(outputMatrix[-1])
        biasMatrix[-1] = delta
        weightMatrix[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            z = outputMatrix[-i]
            sp = self.getDerivativeVal(z)
            # t = self.weights[-i + 1].transpose()
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            biasMatrix[-i] = delta
            weightMatrix[-i] = np.dot(delta, activations[-i - 1].transpose())
        return biasMatrix, weightMatrix

    def getDerivativeVal(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def getError(self, output_activations, y):
        return output_activations - y


if __name__ == "__main__":
    getInputs()
