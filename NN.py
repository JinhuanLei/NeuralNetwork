import networkx as nx
import os
import re

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
    print("length", len(testSet), "  ", len(testLabel))
    print(testSet[0])
    print(testLabel[0])


class Neuron(object):
    def __init__(self, tag, linkedNeurons):
        self.tag = tag
        self.linkedNeurons = linkedNeurons


if __name__ == "__main__":
    getInputs()
