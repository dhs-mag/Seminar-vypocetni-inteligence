import numpy as np
import random as random


class Perceptron:
    def __init__(self, inputs, outputs, activation):
        self.inputs = inputs
        self.outputs = outputs
        self.activation = activation



#
# def sigmoid(x):
#     return np.round(1.0 / (1.0 + np.exp(-x)), 15)
#
#
# alpha = 0.5  # rychlost učení
# eta = 0.1  # setrvačnost
#
#
# class Perceptron:
#     def __init__(self, num_outputs, num_inputs, activation):
#         self.num_inputs = num_inputs
#         self.num_outputs = num_outputs
#         self.weight = np.zeros((num_outputs, num_inputs))
#         self.th = np.zeros(num_outputs)
#         self.activation_function = activation
#         self.delta = np.zeros(num_outputs)
#         self.dws = np.zeros((num_outputs, num_inputs))
#         self.odw = np.zeros((num_outputs, num_inputs))
#         self.dths = np.zeros(num_outputs)
#         self.odth = np.zeros(num_outputs)
#         self.outputs = np.zeros(num_outputs)  # Y outputs
#
#     def output_delta(self, d):
#         diff = d - self.outputs
#         self.delta = diff * (self.outputs * (1 - self.outputs))
#         return diff @ diff / len(self.outputs)
#
#     def learn(self, x):
#         for j in range(len(self.dws)):
#             self.dws[j] += self.delta[j] * x
#         self.dths += -self.delta
#
#     def back_propagate(self, prevLayer):
#         prevLayer.delta = (np.transpose(self.weight) @ self.delta) * (prevLayer.outputs * (1 - prevLayer.outputs))
#
#     def epoch_start(self):
#         self.dws = np.zeros((self.num_outputs, self.num_inputs))
#         self.dths = np.zeros(self.num_outputs)
#         self.outputs = np.zeros(self.num_outputs)
#
#     def epoch_finish(self):
#         dws_temp = alpha * self.dws + eta * self.odw
#         self.weight += dws_temp
#         self.odw = dws_temp
#
#         dths_temp = alpha * self.dths + eta * self.odth
#         self.th += dths_temp
#         self.odth = dths_temp
#
#     def recall(self, inputs_array):
#         self.outputs = self.activation_function(self.weight @ inputs_array - self.th)
#         return self.outputs
#
#     def init(self, random_range_min, random_range_max):
#         for x in range(len(self.weight)):
#             for y in range(len(self.weight[x])):
#                 self.weight[x][y] = random.uniform(random_range_max, random_range_min)
#
#         for x in range(len(self.th)):
#             self.th[x] = random.uniform(random_range_max, random_range_min)
#