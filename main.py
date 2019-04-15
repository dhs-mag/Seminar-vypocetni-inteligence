import numpy as np
import matplotlib.pyplot as plt
import random as random

random.seed(351)


def sigmoida(phi):
    return 1.0 / (1.0 + np.exp(-phi))


class Percepton:
    def __init__(self, num_outputs, num_inputs, activation):

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.activation_function = activation

        # delta pole dana poctem vystupu
        self.delta = np.zeros(num_outputs)

        # vahy
        self.w = np.zeros((num_outputs, num_inputs))
        self.dw = np.zeros((num_outputs, num_inputs))
        # delta w suma
        self.dws = np.zeros((num_outputs, num_inputs))
        # matice stara zmena vahy
        self.odw = np.zeros((num_outputs, num_inputs))

        # theta
        self.th = np.zeros(num_outputs)
        # delta theta
        self.dth = np.zeros(num_outputs)
        # delta theta suma
        self.dths = np.zeros(num_outputs)
        # matice stara zmena prahu th
        self.oth = np.zeros(num_outputs)

        self.outputs = np.zeros(num_outputs)

    def output_delta(self, d):
        # deltai = (di - yi) * (yi * (1 - yi))
        self.delta = (d - self.outputs) * (self.outputs * (1 - self.outputs))
        return (d - self.outputs) @ (d - self.outputs) / len(self.outputs)

    def learn(self, xInputs):
        for i in range(len(self.delta)):
            self.dws[i] += self.delta[i] * xInputs
            self.dths += -self.delta[i]

    def back_propagate(self, prevLayer):
        prevLayer.delta = (np.transpose(self.w) @ self.delta) * (self.outputs * (1 - self.outputs))

    def epoch_start(self):
        self.dws = np.zeros((self.num_outputs, self.num_inputs))
        self.dths = np.zeros(self.num_outputs)
        self.outputs = np.zeros(self.num_outputs)

    def epoch_finish(self):
        a = 1

    def recall(self, inputs_array):
        self.outputs = self.activation_function(self.w @ inputs_array - self.th)
        return self.outputs

    def init(self, randon_range_min, randon_range_max):
        for x in range(len(self.w)):
            for y in range(len(self.w[x])):
                self.w[x][y] = random.uniform(randon_range_max, randon_range_min)

        for x in range(len(self.th)):
            self.th[x] = random.uniform(randon_range_max, randon_range_min)

    # nakresleni oddelovacu
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
        # self.h = Percepton(2, 2, sigmoida)
        # self.o = Percepton(1, 2, sigmoida)
        # self.h.w[0][0] = 8;
        # self.h.w[0][1] = -8;
        # self.h.w[1][0] = 8;
        # self.h.w[1][1] = -8;
        # self.h.th[0] = -4;
        # self.h.th[1] = 4;
        # self.o.w[0][0] = 8;
        # self.o.w[0][1] = -8;
        # self.o.th[0] = 8;

        self.layers = []
        self.output = []

    def test(self):
        z = np.zeros((101, 101))
        for ay in range(101):
            for ax in range(100):
                z[ay, ax] = self.oth.recall(self.th.recall(np.array([ay / 100, ax / 100])))

            lines = net.get_decision_boundaries()

            # Subplots images
            fig, fg = plt.subplots()
            im = fg.imshow(z, interpolation='bilinear', cmap='gray', origin='lower', extent=[0, 1, 0, 1],
                           vmax=1,
                           vmin=0)
            for x in range(len(lines)):
                fg.plot(lines[x][0], lines[x][1])

            fig.colorbar(im)
            plt.xlabel("x0")
            plt.ylabel("x1")
            plt.show()

    def get_decision_boundaries(self):
        result = []
        for x in range(len(self.h.th)):
            result.append(self.decision_boundary(self.h.w[x], self.h.th[x]))
        return result

    def recall(self, x):
        self.layers[0].recall(x)
        return self.layers[1].recall(self.layers[0].outputs)

    def net_init(self, random_range_min, random_range_max):
        self.layers = []
        self.layers.append(Percepton(2, 2, sigmoida))
        self.layers.append(Percepton(1, 2, sigmoida))
        self.output = self.layers[1]

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
        self.layers[1].back_propagate(self.layers[0])

        self.layers[1].learn()
        self.layers[0].learn()
        return e


if __name__ == "__main__":
    net = Net()
    net.net_init(-0.3, 0.3)

    trainSet = [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]],
    ]
    print("Second")
    print(net.recall(trainSet[0][0]))

    net.epoch_start()
    for pat in trainSet:
        net.learn(pat[0], pat[1])
    net.epoch_finish()
