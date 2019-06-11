import random

import numpy as np
from numpy.core.tests.test_mem_overlap import xrange


def sigmoid(x):
    return np.round(1.0 / (1.0 + np.exp(-x)), 15)


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def forward_propagation(self, a):
        """
        a = input
        b = bias

        Returns
        the output of the network

        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def train(self, training_data, epochs, mini_batch_size, eta,
              test_data=None):
        """
        Train the neural network using stochastic gradient descent.

        training_data = list of tuples
        epochs = number of epochs to train
        mini_batch_size = size to sample
        eta = learning rate

        If "test_data" is provided then its tested after each epoch
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_derivative(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.forward_propagation(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return output_activations - y

# class Net:
#     def __init__(self, layer=None):
#         if layer is None:
#             layer = []
#
#         self.layers = layer
#         # self.output = []
#
#     def recall(self, x):
#         self.layers[0].recall(x)
#         return self.layers[1].recall(self.layers[0].outputs)
#
#     def add_neuron(self, neuron: Perceptron):
#         self.layers.append(neuron)
#
#     def net_init(self, random_range_min, random_range_max):
#         # self.hidden_layer = []
#         # self.output = self.hidden_layer[1]
#         for l in self.layers:
#             l.init(random_range_min, random_range_max)
#
#     def epoch_start(self):
#         for l in self.layers:
#             l.epoch_start()
#
#     def epoch_finish(self):
#         for l in self.layers:
#             l.epoch_finish()
#
#     def learn(self, x, d):
#         self.recall(x)
#         e = self.layers[1].output_delta(d)
#         self.layers[1].learn(self.layers[0].outputs)
#         self.layers[1].back_propagate(self.layers[0])
#         self.layers[0].learn(x)
#         return e
#
#     def print_net(self):
#         print("% 1.15f" % self.layers[1].outputs[0] + "; output: y")
#         print("% 1.15f" % self.layers[1].th[0] + "; output: threshold")
#         print("% 1.15f" % self.layers[1].w[0][0] + "; output: w[0]")
#         print("% 1.15f" % self.layers[1].w[0][1] + "; output: w[1]")
#         print("% 1.15f" % self.layers[1].delta[0] + "; output: delta")
#         print("% 1.15f" % self.layers[1].dths[0] + "; output: delta threshold")
#         print("% 1.15f" % self.layers[1].dws[0][0] + "; output: delta w[0]")
#         print("% 1.15f" % self.layers[1].dws[0][1] + "; output: delta w[1]")
#         print("% 1.15f" % self.layers[0].outputs[0] + "; hidden: y[0]")
#         print("% 1.15f" % self.layers[0].outputs[1] + "; hidden: y[1]")
#         print("% 1.15f" % self.layers[0].th[0] + "; hidden: threshold[0]")
#         print("% 1.15f" % self.layers[0].th[1] + "; hidden: threshold[1]")
#         print("% 1.15f" % self.layers[0].w[0][0] + "; hidden: w[0][0]")
#         print("% 1.15f" % self.layers[0].w[0][1] + "; hidden: w[0][1]")
#         print("% 1.15f" % self.layers[0].w[1][0] + "; hidden: w[1][0]")
#         print("% 1.15f" % self.layers[0].w[1][1] + "; hidden: w[1][1]")
#         print("% 1.15f" % self.layers[0].delta[0] + "; hidden: delta[0]")
#         print("% 1.15f" % self.layers[0].delta[1] + "; hidden: delta[1]")
#         print("% 1.15f" % self.layers[0].dths[0] + "; hidden: delta hreshold[0]")
#         print("% 1.15f" % self.layers[0].dths[1] + "; hidden: delta threshold[1]")
#         print("% 1.15f" % self.layers[0].dws[0][0] + "; hidden: delta w[0][0]")
#         print("% 1.15f" % self.layers[0].dws[0][1] + "; hidden: delta w[0][1]")
#         print("% 1.15f" % self.layers[0].dws[1][0] + "; hidden: delta w[1][0]")
#         print("% 1.15f" % self.layers[0].dws[1][1] + "; hidden: delta w[1][1]")
#         print("% 1.15f" % self.layers[0].odw[0][0] + "; hidden: old delta w[0][0]")
#         print("% 1.15f" % self.layers[0].odw[0][1] + "; hidden: old delta w[0][1]")
#         print("% 1.15f" % self.layers[0].odw[1][0] + "; hidden: old delta w[1][0]")
#         print("% 1.15f" % self.layers[0].odw[1][1] + "; hidden: old delta w[1][1]")
