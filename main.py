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
        print('delta', self.delta)
        return (d - self.outputs) @ (d - self.outputs) / len(self.outputs)

    def learn(self, xInputs):
        # for i in range(len(self.delta)):
            self.dws += self.delta * xInputs
            self.dths += -self.delta

    def backPropagate(self, prevLayer):
        print('out', np.transpose(self.w) @ self.delta)
        prevLayer.delta = (np.transpose(self.w) @ self.delta) * (prevLayer.outputs * (1 - prevLayer.outputs))

    def epochStart(self):
        self.dws = np.zeros((self.num_outputs, self.num_inputs))
        self.dths = np.zeros(self.num_outputs)
        self.outputs = np.zeros(self.num_outputs)

    def epochFinish(self, eta, alpha):
        self.dws = eta * self.dws + alpha * self.odw
        self.w += self.dws
        self.odw = self.w

        self.dth = eta * self.dth + alpha * self.oth
        self.th += self.dth
        self.oth = self.th

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

        for l in self.layers:
            l.init(randon_range_min, randon_range_max)

    def epochStart(self):
        for l in self.layers:
            l.epochStart()

    def epochFinish(self, eta, alpha):
        for l in self.layers:
            l.epochFinish(eta, alpha)

    def learn(self, x, d):
        self.recall(x)
        e = self.layers[1].outputDelta(d)
        print('mse', e)
        self.layers[1].learn(self.layers[0].outputs)
        self.layers[1].backPropagate(self.layers[0])
        self.layers[0].learn(x)
        return e


if __name__ == "__main__":
    net = Net()
    net.netInit(-0.3, 0.3)

    trainSet = np.array([
        np.array([np.array([0, 0]), np.array([0])]),
        np.array([np.array([1, 0]), np.array([1])]),
        np.array([np.array([0, 1]), np.array([1])]),
        np.array([np.array([1, 1]), np.array([0])]),
    ])
    print("Before learn:", net.recall(trainSet[0][0]))
    eta = .8
    alpha = .5
    avgErr = 0
    for i in range(100):
        print("Epoch:", i + 1)
        avgErr = 0
        net.epochStart()
        for pat in trainSet:
            print('pat', pat[0])
            avgErr += net.learn(pat[0], pat[1])
        net.epochFinish(eta, alpha)
        print("Error:", avgErr / len(trainSet))
        print("======")

    print("After learn:", net.recall(trainSet[0][0]))
