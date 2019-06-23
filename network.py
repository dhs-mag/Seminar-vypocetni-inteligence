class Network:
    def __init__(self):
        self.layers = []
        self.output_layer = []

    def initialize_network(self, perceptron_layers, random_range_min, random_range_max):
        for perceptron_layer in perceptron_layers:
            self.layers.append(perceptron_layer)
        self.output_layer = self.layers[len(self.layers) - 1]

        for layer in self.layers:
            layer.initialize_perceptron(random_range_min, random_range_max)

        # xor test
        # self.layers[0].weights[0][0] = -0.214767760000000
        # self.layers[0].weights[0][1] = -0.045404790000000
        # self.layers[0].weights[1][0] = 0.106739550000000
        # self.layers[0].weights[1][1] = 0.136999780000000
        # self.layers[0].bias[0] = -0.299236760000000
        # self.layers[0].bias[1] = 0.122603690000000
        # self.layers[1].weights[0][0] = 0.025870070000000
        # self.layers[1].weights[0][1] = 0.168638190000000
        # self.layers[1].bias[0] = 0.019322390000000

    def feed_forward(self, input_array):
        # input layer to hidden
        self.layers[0].calculate_perceptron(input_array)

        # hidden layers
        for layer in range(len(self.layers)):
            if 1 <= layer <= len(self.layers):
                return self.layers[layer].calculate_perceptron(self.layers[layer - 1].outputs)

    def epoch_start(self):
        for layer in self.layers:
            layer.epoch_init_perceptron()

    def learn(self, input_array, desired_output):
        self.feed_forward(input_array)
        error = self.layers[len(self.layers) - 1].output_delta(desired_output)

        array_size = len(self.layers)

        for i in range(array_size - 1, 0, -1):
            if 1 <= i <= array_size:
                self.layers[i].learn_perceptron(self.layers[i - 1].outputs)
                self.layers[i].back_propagate(self.layers[i - 1])
        self.layers[0].learn_perceptron(input_array)
        return error

        # self.layers[1].learn_perceptron(self.layers[0].outputs)
        # self.layers[1].back_propagate(self.layers[0])
        # self.layers[0].learn_perceptron(input_array)
        # return error

    def epoch_finish(self, speed, inertia):
        for layer in self.layers:
            layer.epoch_finish_perceptron(speed, inertia)

    def debug_log(self):
        self.layer_weights_log()
        self.layer_bias_log()

    def layer_weights_log(self):
        count_i = 0
        count_j = 0
        for layer in range(len(self.layers)):
            # layer 0 hidden
            if layer == 0:
                for i in self.layers[layer].weights:
                    for j in i:
                        print("{: 1.15f}".format(j), end='')
                        print(" hidden:w[{:d}][{:d}]".format(count_i, count_j))
                        count_j += 1
                    count_i += 1
                    count_j = 0
                count_i = 0

            # layer 1 output
            if layer >= 1:
                for i in self.layers[layer].weights:
                    for j in i:
                        print("{: 1.15f}".format(j), end='')
                        print(" output:w[{:d}][{:d}]".format(count_i, count_j))
                        count_j += 1
                    count_i += 1
                    count_j = 0
                count_i = 0

    def layer_bias_log(self):
        count_j = 0
        for layer in range(len(self.layers)):
            # layer 0 hidden
            if layer == 0:
                for i in self.layers[layer].bias:
                    print("{: 1.15f}".format(i), end='')
                    print(" hidden:bias[{:d}]".format(count_j))
                    count_j += 1
                count_j = 0

        # layer 1 output
        if layer >= 1:
            for i in self.layers[layer].bias:
                print("{: 1.15f}".format(i), end='')
                print(" output:bias[{:d}]".format(count_j))
                count_j += 1

    def print_net(self):
        print("%1.15f" % self.layers[1].outputs[0] + ";output:y")
        print("%1.15f" % self.layers[1].bias[0] + ";output:threshold")
        print("%1.15f" % self.layers[1].weights[0][0] + ";output:w[0]")
        print("%1.15f" % self.layers[1].weights[0][1] + ";output:w[1]")
        print("%1.15f" % self.layers[1].delta[0] + ";output:delta")
        print("%1.15f" % self.layers[1].dif_bias_sum[0] + ";output:deltathreshold")
        print("%1.15f" % self.layers[1].dif_weight_sum[0][0] + ";output:deltaw[0]")
        print("%1.15f" % self.layers[1].dif_weight_sum[0][1] + ";output:deltaw[1]")
        print("%1.15f" % self.layers[0].outputs[0] + ";hidden:y[0]")
        print("%1.15f" % self.layers[0].outputs[1] + ";hidden:y[1]")
        print("%1.15f" % self.layers[0].bias[0] + ";hidden:threshold[0]")
        print("%1.15f" % self.layers[0].bias[1] + ";hidden:threshold[1]")
        print("%1.15f" % self.layers[0].weights[0][0] + ";hidden:w[0][0]")
        print("%1.15f" % self.layers[0].weights[0][1] + ";hidden:w[0][1]")
        print("%1.15f" % self.layers[0].weights[1][0] + ";hidden:w[1][0]")
        print("%1.15f" % self.layers[0].weights[1][1] + ";hidden:w[1][1]")
        print("%1.15f" % self.layers[0].delta[0] + ";hidden:delta[0]")
        print("%1.15f" % self.layers[0].delta[1] + ";hidden:delta[1]")
        print("%1.15f" % self.layers[0].dif_bias_sum[0] + ";hidden:deltathreshold[0]")
        print("%1.15f" % self.layers[0].dif_bias_sum[1] + ";hidden:deltathreshold[1]")
        print("%1.15f" % self.layers[0].dif_weight_sum[0][0] + ";hidden:deltaw[0][0]")
        print("%1.15f" % self.layers[0].dif_weight_sum[0][1] + ";hidden:deltaw[0][1]")
        print("%1.15f" % self.layers[0].dif_weight_sum[1][0] + ";hidden:deltaw[1][0]")
        print("%1.15f" % self.layers[0].dif_weight_sum[1][1] + ";hidden:deltaw[1][1]")
        print("%1.15f" % self.layers[0].dif_weight_sum[0][0] + ";hidden:olddeltaw[0][0]")
        print("%1.15f" % self.layers[0].dif_weight_sum[0][1] + ";hidden:olddeltaw[0][1]")
        print("%1.15f" % self.layers[0].dif_weight_sum[1][0] + ";hidden:olddeltaw[1][0]")
        print("%1.15f" % self.layers[0].dif_weight_sum[1][1] + ";hidden:olddeltaw[1][1]")
