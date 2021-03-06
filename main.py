import numpy as np
from PIL import Image, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random as random

random.seed(351);

def sigmoida(phi):
    return 1.0 / (1.0 + np.exp(-phi))

class Percepton:
    def __init__(self, num_outputs, num_inputs, activation):
        self.num_inputs = num_inputs;
        self.num_outputs = num_outputs;
        self.w = np.zeros((num_outputs, num_inputs));
        self.th = np.zeros(num_outputs);
        self.activation_function = activation;
        self.delta = np.zeros(num_outputs);
        self.dw = np.zeros((num_outputs, num_inputs));
        self.dws = np.zeros((num_outputs, num_inputs));
        self.odw = np.zeros((num_outputs, num_inputs));
        self.dth = np.zeros(num_outputs);
        self.dths = np.zeros(num_outputs);
        self.oth = np.zeros(num_outputs);
        self.outputs = np.zeros(num_outputs); #Y - outputs from Perceptron

    def outputDelta(self, d):
        # deltai = (di - yi) * (yi * (1 - yi))
        self.delta = (d - self.outputs) * (self.outputs * (1 - self.outputs));
        return (d - self.outputs) @ (d - self.outputs) / len(self.outputs);

    def learn(self, xInputs):
        for i in range(len(self.delta)):
            self.dws[i] += self.delta[i] * xInputs;
            self.dths += -self.delta[i];

    def backPropagate(self, prevLayer):
        prevLayer.delta = (np.transpose(self.w) @ self.delta) * (self.outputs * (1 - self.outputs));


    def epochStart(self):
        self.dws = np.zeros((self.num_outputs, self.num_inputs));
        self.dths = np.zeros(self.num_outputs);
        self.outputs = np.zeros(self.num_outputs);

    def epochFinish(self):
        a = 1;

    def recall(self, inputs_array):
        self.outputs = self.activation_function(self.w @ inputs_array - self.th);
        return self.outputs;

    def init(self, randon_range_min, randon_range_max):
        for x in range(len(self.w)):
            for y in range(len(self.w[x])):
                self.w[x][y] = random.uniform(randon_range_max, randon_range_min)

        for x in range(len(self.th)):
            self.th[x] = random.uniform(randon_range_max, randon_range_min)


def decision_boudary(w, th):
    x0_0 = -th/-w[0];
    x0_1 = (w[1] -th)/-w[0];
    
    x1_0 = -th/-w[1];
    x1_1 = (w[0] -th)/-w[1];

    result = []
    if(x0_0 >= 0 and x0_0 <= 1):
        result.append([0, x0_0]);
    if(x0_1 >= 0 and x0_1 <= 1):
        result.append([1, x0_1]);
    if(x1_0 >= 0 and x1_0 <= 1):
        result.append([ x1_0, 0]);
    if(x1_1 >= 0 and x1_1 <= 1):
        result.append([ x1_1, 1]);
        
    return result;



class Net:
    def __init__(self):
        #self.h = Percepton(2, 2, sigmoida)
        #self.o = Percepton(1, 2, sigmoida)
        #self.h.w[0][0] = 8;
        #self.h.w[0][1] = -8;
        #self.h.w[1][0] = 8;
        #self.h.w[1][1] = -8;
        #self.h.th[0] = -4;
        #self.h.th[1] = 4;
        #self.o.w[0][0] = 8;
        #self.o.w[0][1] = -8;
        #self.o.th[0] = 8;

        self.layers = [];
        self.output = [];


    def test(self):
        size = 101;
        z = np.zeros((size, size));

        for ax in range(size):
            for ay in range(size):
                z[ax, ay] = self.o.recall(self.h.recall(np.array([ax/(size-1), ay/(size-1)])))[0]*255;

        lines = net.get_decision_boudaries();         

        # Subpolots images
        fig, ax = plt.subplots();
        img = ax.imshow(z, interpolation="bilinear", cmap="gray", origin="lower", extent=[0, 1, 0, 1], vmax=1, vmin=0);
        for x in range(len(lines)):
            ax.plot(lines[x][0], lines[x][1])
        plt.xlabel("x1");
        plt.ylabel("x2");
        plt.show();


    def get_decision_boudaries(self):
        result = [];
        for x in range(len(self.h.th)):
            result.append(decision_boudary(self.h.w[x], self.h.th[x]));
        return result;


    def recall(self, x):
        self.layers[0].recall(x)
        return self.layers[1].recall(self.layers[0].outputs)

    def netInit(self, randon_range_min, randon_range_max):
        self.layers = [];
        self.layers.append(Percepton(2, 2, sigmoida));
        self.layers.append(Percepton(1, 2, sigmoida));
        self.output = self.layers[1]

        for l in self.layers:
            l.init(randon_range_min, randon_range_max)

    def epochStart(self):
        for l in self.layers:
            l.epochStart()

    def epochFinish(self):
        for l in self.layers:
            l.epochFinish()

    def learn(self, x, d):
        self.recall(x)
        e = self.layers[1].outputDelta(d)
        self.layers[1].backPropagate(self.layers[0])

        self.layers[1].learn()
        self.layers[0].learn()
        return e
    
    


if __name__ == "__main__":
    net = Net();
    net.netInit(-0.3, 0.3)

    trainSet = [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]],
    ]
    print("Second")
    print(net.recall(trainSet[0][0]))

    net.epochStart()
    for pat in trainSet:
        net.learn(pat[0], pat[1])
    net.epochFinish()