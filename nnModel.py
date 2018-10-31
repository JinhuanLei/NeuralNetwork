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
    # normaliseData(int(resolution))
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
    nn.train(trainData, testData, 1000 )


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
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for (b, w) in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, test_data, epochs):
        for i in range(epochs):
            mini_batches = [training_data[k:k + 1] for k in range(0, len(training_data))]
            for mini_batch in mini_batches:
                # 根据每个小样本来更新 w 和 b，代码在下一段
                self.updateWeight(mini_batch)
            print("epoch ",i)
            print(" training Accurate", self.evaluate(training_data))
            print(" testing Accurate",self.evaluate(test_data))


    def sigmoid(self,z):
        return 1.0 / (1.0 + np.exp(-z))

    def updateWeight(self, data):
        # 根据 biases 和 weights 的行列数创建对应的全部元素值为 0 的空矩阵
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for (x, y) in data:
            x = self.matrixTranspose(x)
            y = self.matrixTranspose(y)
            # 根据样本中的每一个输入 x 的其输出 y，计算 w 和 b 的偏导数
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # 累加储存偏导值 delta_nabla_b 和 delta_nabla_w
            nabla_b = [nb + dnb for (nb, dnb) in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for (nw, dnw) in zip(nabla_w, delta_nabla_w)]
        # 更新根据累加的偏导值更新 w 和 b，这里因为用了小样本，
        # 所以 eta 要除于小样本的长度
        self.weights = [w - (1 / len(data)) * nw
                        for (w, nw) in zip(self.weights, nabla_w)]
        self.biases = [b - (1 / len(data)) * nb
                       for (b, nb) in zip(self.biases, nabla_b)]

    def evaluate(self, test_data):
        test_results = [(self.feedforward(self.matrixTranspose(x)), self.matrixTranspose(y))
                        for (x, y) in test_data]
        count = 0
        for (x,y) in test_results:
            maxVal = x.max()
            maxIndex = (np.where(x == maxVal))[0]
            if y[maxIndex] == 1.0:
                count += 1
        return count/len(test_results)

    def matrixTranspose(self,x):
        x = np.array(x)
        x = x.reshape(-1,1)
        return x

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传输
        x = self.matrixTranspose(x)
        y = self.matrixTranspose(y)
        activation = x
        # 储存每层的神经元的值的矩阵，下面循环会 append 每层的神经元的值
        activations = [x]
        # 储存每个未经过 sigmoid 计算的神经元的值
        zs = []
        for (b, w) in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # 求 δ 的值
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 乘于前一层的输出值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for i in range(2, self.num_layers):
            # 从倒数第 **l** 层开始更新，**-l** 是 python 中特有的语法表示从倒数第 l 层开始计算
            # 下面这里利用 **l+1** 层的 δ 值来计算 **l** 的 δ 值
            z = zs[-i]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sp
            nabla_b[-i] = delta
            nabla_w[-i] = np.dot(delta, activations[-i - 1].transpose())
        return (nabla_b, nabla_w)

    def sigmoid_prime(self,z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

if __name__ == "__main__":
    getInputs()
