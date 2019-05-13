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
        # self.layers[0].w[0][0] = -  0.214767760000000
        # self.layers[0].w[0][1] = -  0.045404790000000
        # self.layers[0].w[1][0] =    0.106739550000000
        # self.layers[0].w[1][1] =    0.136999780000000
        # self.layers[0].th[0]   = -  0.299236760000000
        # self.layers[0].th[1]   =    0.122603690000000
        # self.layers[1].w[0][0] =    0.025870070000000
        # self.layers[1].w[0][1] =    0.168638190000000
        # self.layers[1].th[0]   =    0.019322390000000

    def epochStart(self):
        for l in self.layers:
            l.epochStart()

    def epochFinish(self):
        for l in self.layers:
            l.epochFinish()

    def learn(self, x, d):
        self.recall(x)
        e = self.layers[1].outputDelta(d)
        # print('mse', e)
        self.layers[1].learn(self.layers[0].outputs)
        self.layers[1].backPropagate(self.layers[0])
        self.layers[0].learn(x)
        return e

    def print_net(self, file=sys.stdout):
        print(("  % 1.15f" % self.layers[1].outputs[0]).replace('.', ',') + " ; output: y ", file=file)
        print(("  % 1.15f" % -self.layers[1].th[0]).replace('.', ',') + " ; output: threshold", file=file)
        print(("  % 1.15f" % self.layers[1].w[0][0]).replace('.', ',') + " ; output: w[0]", file=file)
        print(("  % 1.15f" % self.layers[1].w[0][1]).replace('.', ',') + " ; output: w[1]", file=file)
        print(("  % 1.15f" % self.layers[1].delta[0]).replace('.', ',') + " ; output: delta", file=file)
        print(("  % 1.15f" % -self.layers[1].dths[0]).replace('.', ',') + " ; output: delta threshold", file=file)
        print(("  % 1.15f" % self.layers[1].dws[0][0]).replace('.', ',') + " ; output: delta w[0]", file=file)
        print(("  % 1.15f" % self.layers[1].dws[0][1]).replace('.', ',') + " ; output: delta w[1]", file=file)
        print(("  % 1.15f" % self.layers[0].outputs[0]).replace('.', ',') + " ; hidden: y[0]", file=file)
        print(("  % 1.15f" % self.layers[0].outputs[1]).replace('.', ',') + " ; hidden: y[1]", file=file)
        print(("  % 1.15f" % -self.layers[0].th[0]).replace('.', ',') + " ; hidden: threshold [0]", file=file)
        print(("  % 1.15f" % -self.layers[0].th[1]).replace('.', ',') + " ; hidden: threshold [1]", file=file)
        print(("  % 1.15f" % self.layers[0].w[0][0]).replace('.', ',') + " ; hidden: w[0][0]", file=file)
        print(("  % 1.15f" % self.layers[0].w[0][1]).replace('.', ',') + " ; hidden: w[0][1]", file=file)
        print(("  % 1.15f" % self.layers[0].w[1][0]).replace('.', ',') + " ; hidden: w[1][0]", file=file)
        print(("  % 1.15f" % self.layers[0].w[1][1]).replace('.', ',') + " ; hidden: w[1][1]", file=file)
        print(("  % 1.15f" % self.layers[0].delta[0]).replace('.', ',') + " ; hidden: delta [0]", file=file)
        print(("  % 1.15f" % self.layers[0].delta[1]).replace('.', ',') + " ; hidden: delta [1]", file=file)
        print(("  % 1.15f" % -self.layers[0].dths[0]).replace('.', ',') + " ; hidden: delta threshold [0]", file=file)
        print(("  % 1.15f" % -self.layers[0].dths[1]).replace('.', ',') + " ; hidden: delta threshold [1]", file=file)
        print(("  % 1.15f" % self.layers[0].dws[0][0]).replace('.', ',') + " ; hidden: delta w[0][0]", file=file)
        print(("  % 1.15f" % self.layers[0].dws[0][1]).replace('.', ',') + " ; hidden: delta w[0][1]", file=file)
        print(("  % 1.15f" % self.layers[0].dws[1][0]).replace('.', ',') + " ; hidden: delta w[1][0]", file=file)
        print(("  % 1.15f" % self.layers[0].dws[1][1]).replace('.', ',') + " ; hidden: delta w[1][1]", file=file)
        print(("  % 1.15f" % -self.layers[0].odth[0]).replace('.', ',') + " ; hidden: old delta threshold [0]", file=file)
        print(("  % 1.15f" % -self.layers[0].odth[1]).replace('.', ',') + " ; hidden: old delta threshold [1]", file=file)
        print(("  % 1.15f" % self.layers[0].odw[0][0]).replace('.', ',') + " ; hidden: old delta w[0][0]", file=file)
        print(("  % 1.15f" % self.layers[0].odw[0][1]).replace('.', ',') + " ; hidden: old delta w[0][1]", file=file)
        print(("  % 1.15f" % self.layers[0].odw[1][0]).replace('.', ',') + " ; hidden: old delta w[1][0]", file=file)
        print(("  % 1.15f" % self.layers[0].odw[1][1]).replace('.', ',') + " ; hidden: old delta w[1][1]", file=file)
