import numpy as np
import random as random

random.seed(0)


class Perceptron:
    def __init__(self, num_inputs, num_outputs, activation):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.weights = np.zeros((num_outputs, num_inputs))
        self.bias = np.zeros(num_outputs)  # theta
        self.activation_function = activation

        self.outputs = np.zeros(num_outputs)  # result of Perceptron

        # difference sums
        self.delta = np.zeros(num_outputs)
        self.dif_weight_sum = np.zeros((num_outputs, num_inputs))
        self.dif_bias_sum = np.zeros(num_outputs)

        # old values
        self.old_dif_weight = np.zeros((num_outputs, num_inputs))
        self.old_dif_bias = np.zeros(num_outputs)

    def initialize_perceptron(self, random_range_min, random_range_max):
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] = random.uniform(random_range_min, random_range_max)

        for x in range(len(self.bias)):
            self.bias[x] = random.uniform(random_range_min, random_range_max)

    def calculate_perceptron(self, input_array):
        self.outputs = self.activation_function(self.weights @ input_array - self.bias)
        return self.outputs

    def epoch_init_perceptron(self):
        self.dif_weight_sum = np.zeros((self.num_outputs, self.num_inputs))
        self.dif_bias_sum = np.zeros(self.num_outputs)
        self.outputs = np.zeros(self.num_outputs)

    # doresit
    def output_delta(self, desired_output):
        diff = desired_output - self.outputs
        self.delta = diff * (self.outputs * (1 - self.outputs))

        return diff @ diff / len(self.outputs)

    def learn_perceptron(self, array):
        for i in range(len(self.dif_weight_sum)):
            self.dif_weight_sum[i] += self.delta[i] * array
        self.dif_bias_sum += -self.delta

    def back_propagate(self, previous_layer):
        previous_layer.delta = (np.transpose(self.weights) @ self.delta) * (
                    previous_layer.outputs * (1 - previous_layer.outputs))

    def epoch_finish_perceptron(self, speed, inertia):
        dif_weight_sum_temp = speed * self.dif_weight_sum + inertia * self.old_dif_weight
        self.weights += dif_weight_sum_temp
        self.old_dif_weight = dif_weight_sum_temp

        dif_bias_temp = speed * self.dif_bias_sum + inertia * self.old_dif_bias
        self.bias += dif_bias_temp
        self.old_dif_bias = dif_bias_temp
