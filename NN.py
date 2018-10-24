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
                return
    else:
        print("Goodbye.")
        return


getInputs()


class Neuron(object):
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
