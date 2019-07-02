import sys

from PerceptronLayer import PerceptonLayer, sigmoida


class Net:
    def __init__(self):
        self.layers = []
        self.output = []

    def recall(self, x):
        self.layers[0].recall(x)
        return self.layers[1].recall(self.layers[0].outputs)

    def netInit(self, randon_range_min, randon_range_max):
        self.layers = []
        self.layers.append(PerceptonLayer(25, 25, sigmoida))
        self.layers.append(PerceptonLayer(20, 25, sigmoida))
        self.output = self.layers[1]
        for l in self.layers:
            l.init(randon_range_min, randon_range_max)

    def learn(self, x, d):
        self.recall(x)
        e = self.layers[1].outputDelta(d)
        # print('mse', e)
        self.layers[1].learn(self.layers[0].outputs)
        self.layers[1].backPropagate(self.layers[0])
        self.layers[0].learn(x)
        return e
