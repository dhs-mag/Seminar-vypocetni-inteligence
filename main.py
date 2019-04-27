# import matplotlib.pyplot as plt
import numpy as np
import random as random

random.seed(351)


def sigmoida(phi):
    return 1.0 / (1.0 + np.exp(-phi))


class Percepton:
    def __init__(self, num_outputs, num_inputs, activation):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.w = np.zeros((num_outputs, num_inputs))
        self.th = np.zeros(num_outputs)
        self.activation_function = activation
        self.delta = np.zeros(num_outputs)
        self.dw = np.zeros((num_outputs, num_inputs))
        self.dws = np.zeros((num_outputs, num_inputs))
        self.odw = np.zeros((num_outputs, num_inputs))
        self.dth = np.zeros(num_outputs)
        self.dths = np.zeros(num_outputs)
        self.oth = np.zeros(num_outputs)
        self.outputs = np.zeros(num_outputs)  # Y - outputs from Perceptron

    def outputDelta(self, d):
        # deltai = (di - yi) * (yi * (1 - yi))
        self.delta = (d - self.outputs) * (self.outputs * (1 - self.outputs))
        # print('delta', self.delta)
        return (d - self.outputs) @ (d - self.outputs) / len(self.outputs)

    def learn(self, xInputs):
        # for i in range(len(self.delta)):
            self.dws += self.delta * xInputs
            self.dths -= self.delta

    def backPropagate(self, prevLayer):
        # print('out', np.transpose(self.w) @ self.delta)
        prevLayer.delta = (np.transpose(self.w) @ self.delta) * (prevLayer.outputs * (1 - prevLayer.outputs))

    def epochStart(self):
        self.dws = np.zeros((self.num_outputs, self.num_inputs))
        self.dths = np.zeros(self.num_outputs)
        self.outputs = np.zeros(self.num_outputs)

    def epochFinish(self, eta, alpha):
        self.dws = eta * self.dws + alpha * self.odw
        self.odw = self.dws
        self.w += self.dws

        self.dths = eta * self.dths + alpha * self.oth
        self.oth = self.dths
        self.th += self.dths

    def recall(self, inputs_array):
        self.outputs = self.activation_function(self.w @ inputs_array - self.th)
        return self.outputs

    def init(self, randon_range_min, randon_range_max):
        for x in range(len(self.w)):
            for y in range(len(self.w[x])):
                self.w[x][y] = random.uniform(randon_range_max, randon_range_min)

        for x in range(len(self.th)):
            self.th[x] = random.uniform(randon_range_max, randon_range_min)


def decision_boudary(w, th):
    x0_0 = -th / -w[0]
    x0_1 = (w[1] - th) / -w[0]

    x1_0 = -th / -w[1]
    x1_1 = (w[0] - th) / -w[1]

    result = []
    if (x0_0 >= 0 and x0_0 <= 1):
        result.append([0, x0_0])
    if (x0_1 >= 0 and x0_1 <= 1):
        result.append([1, x0_1])
    if (x1_0 >= 0 and x1_0 <= 1):
        result.append([x1_0, 0])
    if (x1_1 >= 0 and x1_1 <= 1):
        result.append([x1_1, 1])

    return result


class Net:
    def __init__(self):
        self.layers = []
        self.output = []

    def recall(self, x):
        self.layers[0].recall(x)
        return self.layers[1].recall(self.layers[0].outputs)

    def netInit(self, randon_range_min, randon_range_max):
        self.layers = []
        self.layers.append(Percepton(2, 2, sigmoida))
        self.layers.append(Percepton(1, 2, sigmoida))
        self.output = self.layers[1]
        self.layers[0].w[0][0] = -  0.214767760000000
        self.layers[0].w[0][1] = -  0.045404790000000
        self.layers[0].w[1][0] =    0.106739550000000
        self.layers[0].w[1][1] =    0.136999780000000
        self.layers[0].th[0]   = -  0.299236760000000
        self.layers[0].th[1]   =    0.122603690000000
        self.layers[1].w[0][0] =    0.025870070000000
        self.layers[1].w[0][1] =    0.168638190000000
        self.layers[1].th[0]   =    0.019322390000000
        # for l in self.layers:
            # l.init(randon_range_min, randon_range_max)

    def epochStart(self):
        for l in self.layers:
            l.epochStart()

    def epochFinish(self, eta, alpha):
        for l in self.layers:
            l.epochFinish(eta, alpha)

    def learn(self, x, d):
        self.recall(x)
        e = self.layers[1].outputDelta(d)
        # print('mse', e)
        self.layers[1].learn(self.layers[0].outputs)
        self.layers[1].backPropagate(self.layers[0])
        self.layers[0].learn(x)
        return e

    def print_net(self):
        print("%1.15f" % self.layers[1].outputs[0], " ; output: y")
        print("%1.15f" % self.layers[1].th[0], " ; output: treshold")
        print("%1.15f" % self.layers[1].w[0][0], " ; output: w[0]")
        print("%1.15f" % self.layers[1].w[0][1], " ; output: w[1]")
        print("%1.15f" % self.layers[1].delta[0], " ; output: delta")
        print("%1.15f" % self.layers[1].dths[0], " ; output: delta treshold")
        print("%1.15f" % self.layers[1].dws[0][0], " ; output: delta w[0]")
        print("%1.15f" % self.layers[1].dws[0][1], " ; output: delta w[1]")
        print("%1.15f" % self.layers[0].outputs[0], " ; hidden: y[0]")
        print("%1.15f" % self.layers[0].outputs[1], " ; hidden: y[1]")
        print("%1.15f" % self.layers[0].th[0], " ; hidden: treshold [0]")
        print("%1.15f" % self.layers[0].th[1], " ; hidden: treshold [1]")
        print("%1.15f" % self.layers[0].w[0][0], " ; hidden: w[0][0]")
        print("%1.15f" % self.layers[0].w[0][1], " ; hidden: w[0][1]")
        print("%1.15f" % self.layers[0].w[1][0], " ; hidden: w[1][0]")
        print("%1.15f" % self.layers[0].w[1][1], " ; hidden: w[1][1]")
        print("%1.15f" % self.layers[0].delta[0], " ; hidden: delta[0]")
        print("%1.15f" % self.layers[0].delta[1], " ; hidden: delta[1]")
        print("%1.15f" % self.layers[0].dths[0], " ; hidden: delta treshold [0]")
        print("%1.15f" % self.layers[0].dths[1], " ; hidden: delta treshold [1]")
        print("%1.15f" % self.layers[0].dws[0][0], " ; hidden: delta w[0][0]")
        print("%1.15f" % self.layers[0].dws[0][1], " ; hidden: delta w[0][1]")
        print("%1.15f" % self.layers[0].dws[1][0], " ; hidden: delta w[1][0]")
        print("%1.15f" % self.layers[0].dws[1][1], " ; hidden: delta w[1][1]")
        print("%1.15f" % self.layers[0].oth[0], " ; hidden: old delta threshold [0]")
        print("%1.15f" % self.layers[0].oth[1], " ; hidden: old delta threshold [1]")
        print("%1.15f" % self.layers[0].odw[0][0], " ; hidden: old delta w[0][0]")
        print("%1.15f" % self.layers[0].odw[0][1], " ; hidden: old delta w[0][1]")
        print("%1.15f" % self.layers[0].odw[1][0], " ; hidden: old delta w[1][0]")
        print("%1.15f" % self.layers[0].odw[1][1], " ; hidden: old delta w[1][1]")

if __name__ == "__main__":
    net = Net()
    net.netInit(-0.3, 0.3)

    trainSet = np.array([
        np.array([np.array([0, 0]), np.array([0])]),
        np.array([np.array([0, 1]), np.array([1])]),
        np.array([np.array([1, 0]), np.array([1])]),
        np.array([np.array([1, 1]), np.array([0])]),
    ])
    print("Before learn:", net.recall(trainSet[0][0]))
    eta = .80000000
    alpha = .500000000
    avgErr = 0
    for i in range(100):
        print("Epoch:", i + 1)
        avgErr = 0
        net.epochStart()
        j = 0
        for pat in trainSet:
            print('iterace ', j+1)
            avgErr += net.learn(pat[0], pat[1])
            net.print_net()
        net.epochFinish(eta, alpha)
        print("weight update")
        net.print_net()
        print("Error:", avgErr / len(trainSet))
        print("======")

    print("After learn:", net.recall(trainSet[0][0]))
