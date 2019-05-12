import csv
import numpy as np
import matplotlib.pyplot as plot
import random as random


def sigmoid(x):
    return np.round(1.0 / (1.0 + np.exp(-x)), 15)


alpha = 0.5  # rychlost učení
eta = 0.1  # setrvačnost


class Perceptron:
    def __init__(self, num_outputs, num_inputs, activation):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.weight = np.zeros((num_outputs, num_inputs))
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
        self.dths += -self.delta

    def back_propagate(self, prevLayer):
        prevLayer.delta = (np.transpose(self.weight) @ self.delta) * (prevLayer.outputs * (1 - prevLayer.outputs))

    def epoch_start(self):
        self.dws = np.zeros((self.num_outputs, self.num_inputs))
        self.dths = np.zeros(self.num_outputs)
        self.outputs = np.zeros(self.num_outputs)

    def epoch_finish(self):
        dws_temp = alpha * self.dws + eta * self.odw
        self.weight += dws_temp
        self.odw = dws_temp

        dths_temp = alpha * self.dths + eta * self.odth
        self.th += dths_temp
        self.odth = dths_temp

    def recall(self, inputs_array):
        self.outputs = self.activation_function(self.weight @ inputs_array - self.th)
        return self.outputs

    def init(self, random_range_min, random_range_max):
        for x in range(len(self.weight)):
            for y in range(len(self.weight[x])):
                self.weight[x][y] = random.uniform(random_range_max, random_range_min)

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
    def __init__(self, layer=None):
        if layer is None:
            layer = []

        self.layers = layer
        # self.output = []

    def recall(self, x):
        self.layers[0].recall(x)
        return self.layers[1].recall(self.layers[0].outputs)

    def add_neuron(self, neuron: Perceptron):
        self.layers.append(neuron)

    def net_init(self, random_range_min, random_range_max):
        # self.hidden_layer = []
        # self.output = self.hidden_layer[1]
        for l in self.layers:
            l.init(random_range_min, random_range_max)

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


def main():
    net = Net()

    net.add_neuron(Perceptron(4, 4, sigmoid))
    net.add_neuron(Perceptron(3, 4, sigmoid))

    net.net_init(-0.3, 0.3)

    filename = 'iris.csv'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    x = list(reader)

    train_set = []

    for item in x:
        if len(item) > 0:
            train_set.append(
                [
                    np.array([
                        normalize(float(item[0]), 4.3, 7.9),
                        normalize(float(item[1]), 2, 4.4),
                        normalize(float(item[2]), 1, 6.9),
                        normalize(float(item[3]), 0.1, 2.5)
                    ]),
                    np.array([
                        1 if item[4] == "Iris-virginica" else 0,
                        1 if item[4] == "Iris-versicolor" else 0,
                        1 if item[4] == "Iris-setosa" else 0
                    ])
                ])

    errors = []
    epoch = []

    for i in range(10000):
        avg_error = 0
        net.epoch_start()
        for pat in train_set:
            avg_error += net.learn(pat[0], pat[1])
        net.epoch_finish()
        error = avg_error / len(train_set)
        errors.append(error)
        epoch.append(i + 1)

        if i % 100 == 0:
            print("EPOCH:", i + 1)
            print("Error:", error)
            print("----------------------------")
            if error < 0.009:
                break

    # graph
    figure, axis = plot.subplots(1, 1)
    axis.plot(epoch, errors)
    axis.set_xlabel('Epoch')
    axis.set_ylabel('Error')
    axis.grid(True)
    plot.show()

    print("After learn " + str(train_set[0][1]) + " :", np.round(net.recall(train_set[0][0])))
    print("After learn " + str(train_set[55][1]) + " :", np.round(net.recall(train_set[55][0])))
    print("After learn " + str(train_set[107][1]) + " :", np.round(net.recall(train_set[107][0])))
    print("After learn " + str(train_set[142][1]) + " :", np.round(net.recall(train_set[142][0])))


if __name__ == "__main__":
    main()
