import csv

import numpy as np
import matplotlib.pyplot as plt
import random as random

random.seed(351)


def sigmoida(phi):
    return np.round(1.0 / (1.0 + np.exp(-phi)), 15)


speed = 0.8  # rychlost učení
inertia = 0.5  # setrvačnost


class Percepton:
    def __init__(self, num_outputs, num_inputs, activation):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.w = np.zeros((num_outputs, num_inputs))
        self.th = np.zeros(num_outputs)
        self.activation_function = activation
        self.delta = np.zeros(num_outputs)
        self.dws = np.zeros((num_outputs, num_inputs))
        self.odw = np.zeros((num_outputs, num_inputs))
        self.dths = np.zeros(num_outputs)
        self.odth = np.zeros(num_outputs)
        self.outputs = np.zeros(num_outputs)  # Y outputs

    def output_delta(self, d):
        diff = d - self.outputs
        self.delta = diff * (self.outputs * (1 - self.outputs))
        return diff @ diff / len(self.outputs)

    def learn(self, x):
        for j in range(len(self.dws)):
            self.dws[j] += self.delta[j] * x

    def back_propagate(self, prevLayer):
        prevLayer.delta = (np.transpose(self.w) @ self.delta) * (prevLayer.outputs * (1 - prevLayer.outputs))

    def epoch_start(self):
        self.dws = np.zeros((self.num_outputs, self.num_inputs))
        self.dths = np.zeros(self.num_outputs)
        self.outputs = np.zeros(self.num_outputs)

    def epoch_finish(self):
        dws_temp = speed * self.dws + inertia * self.odw
        self.w += dws_temp
        self.odw = dws_temp

        dths_temp = speed * self.dths + inertia * self.odth
        self.th += dths_temp
        self.odth = dths_temp

    def recall(self, inputs_array):
        self.outputs = self.activation_function(self.w @ inputs_array - self.th)
        return self.outputs

    def init(self, random_range_min, random_range_max):
        for x in range(len(self.w)):
            for y in range(len(self.w[x])):
                self.w[x][y] = random.uniform(random_range_max, random_range_min)

        for x in range(len(self.th)):
            self.th[x] = random.uniform(random_range_max, random_range_min)

    def decision_boundary(self, w, th):
        x0_0 = -th / -w[0]
        x0_1 = (w[1] - th) / -w[0]

        x1_0 = -th / -w[1]
        x1_1 = (w[0] - th) / -w[1]

        result = []
        # rovnice 1
        if 0 <= x0_0 <= 1:
            result.append([0, x0_0])
        # rovnice 2
        if 0 <= x0_1 <= 1:
            result.append([1, x0_1])
        # rovnice 3
        if 0 <= x1_0 <= 1:
            result.append([x1_0, 0])
        # rovnice 4
        if 0 <= x1_1 <= 1:
            result.append([x1_1, 1])

        return result


class Net:
    def __init__(self):
        self.layers = []
        self.output = []

    # def test(self):
    #     z = np.zeros((101, 101))
    #     for ay in range(101):
    #         for ax in range(100):
    #             z[ay, ax] = self.oth.recall(self.th.recall(np.array([ay / 100, ax / 100])))
    #
    #         lines = net.get_decision_boundaries()
    #
    #         # Subplots images
    #         fig, fg = plt.subplots()
    #         im = fg.imshow(z, interpolation='bilinear', cmap='gray', origin='lower', extent=[0, 1, 0, 1],
    #                        vmax=1,
    #                        vmin=0)
    #         for x in range(len(lines)):
    #             fg.plot(lines[x][0], lines[x][1])
    #
    #         fig.colorbar(im)
    #         plt.xlabel("x0")
    #         plt.ylabel("x1")
    #         plt.show()

    # def get_decision_boundaries(self):
    #     result = []
    #     for x in range(len(self.h.th)):
    #         result.append(self.decision_boundary(self.h.w[x], self.h.th[x]))
    #     return result

    def recall(self, x):
        self.layers[0].recall(x)
        return self.layers[1].recall(self.layers[0].outputs)

    def net_init(self, random_range_min, random_range_max):
        self.layers = []
        self.layers.append(Percepton(4, 4, sigmoida))
        self.layers.append(Percepton(3, 4, sigmoida))
        self.output = self.layers[1]
        for l in self.layers:
            l.init(random_range_min, random_range_max)

        # self.layers = []
        # self.layers.append(Percepton(2, 2, sigmoida))
        # self.layers.append(Percepton(1, 2, sigmoida))
        # self.output = self.layers[1]
        # for l in self.layers:
        #     l.init(random_range_min, random_range_max)
        #
        # # inicializace vah
        # self.layers[0].w[0][0] = -0.214767760000000
        # self.layers[0].w[0][1] = -0.045404790000000
        # self.layers[0].w[1][0] = 0.106739550000000
        # self.layers[0].w[1][1] = 0.136999780000000
        # self.layers[1].w[0][0] = 0.025870070000000
        # self.layers[1].w[0][1] = 0.168638190000000
        #
        # # inicializace threshold
        # self.layers[0].th[0] = -0.299236760000000
        # self.layers[0].th[1] = 0.122603690000000
        # self.layers[1].th[0] = 0.019322390000000

    def epoch_start(self):
        for l in self.layers:
            l.epoch_start()

    def epoch_finish(self):
        for l in self.layers:
            l.epoch_finish()

    def learn(self, x, d):
        self.recall(x)
        e = self.layers[1].output_delta(d)
        self.layers[1].learn(self.layers[0].outputs)
        self.layers[1].back_propagate(self.layers[0])
        self.layers[0].learn(x)
        return e

    def print_net(self):
        print("# iterace ")
        # print("% 1.15f" % self.layers[0].num_inputs[0] + "; input [0]")
        # print("% 1.15f" % self.layers[0].num_inputs[1] + "; input [1]")
        print("% 1.15f" % self.layers[1].outputs[0] + "; output: y")
        print("% 1.15f" % self.layers[1].th[0] + "; output: threshold")
        print("% 1.15f" % self.layers[1].w[0][0] + "; output: w[0]")
        print("% 1.15f" % self.layers[1].w[0][1] + "; output: w[1]")
        print("% 1.15f" % self.layers[1].delta[0] + "; output: delta")
        print("% 1.15f" % self.layers[1].dths[0] + "; output: delta threshold")
        print("% 1.15f" % self.layers[1].dws[0][0] + "; output: delta w[0]")
        print("% 1.15f" % self.layers[1].dws[0][1] + "; output: delta w[1]")
        print("% 1.15f" % self.layers[0].outputs[0] + "; hidden: y[0]")
        print("% 1.15f" % self.layers[0].outputs[1] + "; hidden: y[1]")
        print("% 1.15f" % self.layers[0].th[0] + "; hidden: threshold[0]")
        print("% 1.15f" % self.layers[0].th[1] + "; hidden: threshold[1]")
        print("% 1.15f" % self.layers[0].w[0][0] + "; hidden: w[0][0]")
        print("% 1.15f" % self.layers[0].w[0][1] + "; hidden: w[0][1]")
        print("% 1.15f" % self.layers[0].w[1][0] + "; hidden: w[1][0]")
        print("% 1.15f" % self.layers[0].w[1][1] + "; hidden: w[1][1]")
        print("% 1.15f" % self.layers[0].delta[0] + "; hidden: delta[0]")
        print("% 1.15f" % self.layers[0].delta[1] + "; hidden: delta[1]")
        print("% 1.15f" % self.layers[0].dths[0] + "; hidden: delta hreshold[0]")
        print("% 1.15f" % self.layers[0].dths[1] + "; hidden: delta threshold[1]")
        print("% 1.15f" % self.layers[0].dws[0][0] + "; hidden: delta w[0][0]")
        print("% 1.15f" % self.layers[0].dws[0][1] + "; hidden: delta w[0][1]")
        print("% 1.15f" % self.layers[0].dws[1][0] + "; hidden: delta w[1][0]")
        print("% 1.15f" % self.layers[0].dws[1][1] + "; hidden: delta w[1][1]")
        print("% 1.15f" % self.layers[0].odw[0][0] + "; hidden: old delta w[0][0]")
        print("% 1.15f" % self.layers[0].odw[0][1] + "; hidden: old delta w[0][1]")
        print("% 1.15f" % self.layers[0].odw[1][0] + "; hidden: old delta w[1][0]")
        print("% 1.15f" % self.layers[0].odw[1][1] + "; hidden: old delta w[1][1]")


def normalize(value, min, max):
    return (value - min) / (max - min)


if __name__ == "__main__":
    net = Net()
    net.net_init(-0.3, 0.3)

    filename = 'iris.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    avgErr = 0
    err = 0

    trainSet = []

    # Coding
    # Virginica     100
    # Versicolor    010
    # Setosa        001

    for item in x:
        if len(item) > 0:
            trainSet.append(
                [
                    np.array([
                        normalize(float(item[0]), 4.3, 7.9),
                        normalize(float(item[1]), 2, 4.4),
                        normalize(float(item[2]), 1, 6.9),
                        normalize(float(item[3]), 0.1, 2.5)
                    ]),
                    np.array([
                        1 if item[4] == 100 else 0,
                        1 if item[4] == 010 else 0,
                        1 if item[4] == 001 else 0
                    ])
                ])

    rms = []
    epoch = []
    avgErr = 0
    err = 0

    for i in range(10000):
        avgErr = 0
        net.epoch_start()
        for pat in trainSet:
            avgErr += net.learn(pat[0], pat[1])
        net.epoch_finish()
        err = avgErr / len(trainSet)

        if err < 0.009:
            print("EPOCH:", i + 1)
            print("Error:", err)
            print("----------------------------")
            break

        rms.append(err)
        epoch.append(i + 1)

        if i % 100 == 0:
            print("EPOCH:", i + 1)
            print("Error:", err)
            print("----------------------------")

    # graph
    fig, axs = plt.subplots(1, 1)
    axs.plot(epoch, rms)
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Error')
    axs.grid(True)
    plt.show()

    print("After learn " + str(trainSet[0][1]) + " :", np.round(net.recall(trainSet[0][0])))
    print("After learn " + str(trainSet[55][1]) + " :", np.round(net.recall(trainSet[55][0])))
    print("After learn " + str(trainSet[107][1]) + " :", np.round(net.recall(trainSet[107][0])))
    print("After learn " + str(trainSet[142][1]) + " :", np.round(net.recall(trainSet[142][0])))
