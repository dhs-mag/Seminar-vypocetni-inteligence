
import random
import math

import statistics


class ThreeLayerNeuralNetwork:
    """
    Universal MLP feed-forward neural network with two hidden layers.

    Args:
        nLayer1: Number of neurons in first hidden layer
        nLayer2: Number of neurons in second hidden layer
        nInput: Number of inputs - how many features are there
        nOutput: Number of outputs - how many classes we want to detect
    """

    def __init__(self, nInput, nLayer1, nLayer2, nOutput):
        self.GAMMA = 1.0

        self.nInput = nInput
        self.nLayer1 = nLayer1
        self.nLayer2 = nLayer2
        self.nOutput = nOutput
        self.weights1 = [[0 for i in range(nLayer1)] for j in range(nInput+1)]
        self.weights2 = [[0 for i in range(nLayer2)] for j in range(nLayer1+1)]
        self.weights3 = [[0 for i in range(nOutput)] for j in range(nLayer2+1)]
        self.dWeights1 = [[0 for i in range(nLayer1)] for j in range(nInput+1)]
        self.dWeights2 = [[0 for i in range(nLayer2)] for j in range(nLayer1+1)]
        self.dWeights3 = [[0 for i in range(nOutput)] for j in range(nLayer2+1)]

    def randomizeWeights(self):
        """
        Randomize weights, use before training.
        """
        self.randomizeWeightsMatrix(self.weights1)
        self.randomizeWeightsMatrix(self.weights2)
        self.randomizeWeightsMatrix(self.weights3)


    def randomizeWeightsMatrix(self, weights):
        """
        Randomize weights matrix.

        Individual elements will be random valued from interval <-1,1>

        Args:
            weights: Matrix to be randomized.
        """
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = random.uniform(-1.0, 1.0)

    def recall(self, input):
        """
        Performs recall.

        Args:
            input: Input data vector

        Returns:
            Output vector
        """
        outputLayer1 = [0 for i in range(self.nLayer1)]
        outputLayer2 = [0 for i in range(self.nLayer2)]
        return self.recallProcess(input, outputLayer1, outputLayer2)

    def recallProcess(self, input, outputLayer1, outputLayer2):
        """
        Performs recall using supplied intermediate vectors.

        Args:
            input: Input data vector
            outputLayer1: Vector of first hidden layer (out)
            outputLayer2: Vector of second hidden layer (out)

        Returns:
            Output vector
        """
        output = [0 for i in range(self.nOutput)]
        self.processLayer(input, outputLayer1, self.weights1)
        self.processLayer(outputLayer1, outputLayer2, self.weights2)
        self.processLayer(outputLayer2, output, self.weights3)
        return output

    def processLayer(self, input, outputLayer, weights):
        """
        Process data transit from one neural layer to another.

        Args:
            input: Input layer data vector.
            outputLayer: Output layer data vector.
            weights: Weights between layer vectors, weights matrix.

        """

        for i in range(len(outputLayer)):
            # process virtual bias input
            outputLayer[i] = 1.0 * weights[len(input)][i]

            for j in range(len(input)):
                outputLayer[i] += input[j] * weights[j][i]

            outputLayer[i] *= -self.GAMMA
            outputLayer[i] = 1.0/(1.0 + math.exp(outputLayer[i]))

    def learn(self, input, expectedOutput, learnSpeed, momentum):
        """
        Train neural network using back-propagation method.

        Args:
            input: Array of input data vectors
            expectedOutput: Array of expected (desired) output data vectors
            learnSpeed: Learning speed, optimal is about 0.2
            momentum: Learning momentum, optimal is about 0.8

        Returns:
            Mean error of recall before training
        """

        outputLayer1 = [0 for i in range(self.nLayer1)]
        outputLayer2 = [0 for i in range(self.nLayer2)]

        # test recall
        outputLayer3 = self.recallProcess(input, outputLayer1, outputLayer2)

        # VYSTUPNI VRSTVA
        delta = [self.GAMMA * (outputLayer3[i]*(1-outputLayer3[i])*(expectedOutput[i]-outputLayer3[i])) for i in range(self.nOutput)]

        dWeights = [[learnSpeed*delta[j]*outputLayer2[i] for j in range(self.nOutput)] for i in range(self.nLayer2)]

        dWeights.append([learnSpeed*delta[j] for j in range(self.nOutput)])

        self.plus(self.weights3, dWeights, 1.0)
        self.plus(self.weights3, self.dWeights3, momentum)
        self.dWeights3 = dWeights

        # DRUHA SKRYA VRSTVA

        # 𝑛Σ︁𝑜𝑢𝑡_𝑗=1 𝛿^𝑜𝑢𝑡_𝑘 𝑤^𝑜𝑢𝑡_𝑘𝑗
        sum = [0.0 for i in range(self.nLayer2)]
        for i in range(self.nLayer2):
            for j in range(self.nOutput):
                sum[i]+= delta[j] * self.weights3[i][j]

        # 𝛿^ℎ2_𝑘 = 𝛾 · 𝑦^ℎ2_𝑘 · (1 − 𝑦^ℎ2_𝑘 ) · (𝑛_𝑜𝑢𝑡Σ︁_𝑗=1 𝛿^𝑜𝑢𝑡_𝑘 𝑤^𝑜𝑢𝑡_𝑘𝑗 )
        delta = [self.GAMMA*(outputLayer2[i]*(1-outputLayer2[i])*sum[i]) for i in range(self.nLayer2)]

        # Δ𝑤^ℎ2_𝑖𝑘 = 𝜂 · 𝛿^ℎ2_𝑘 · 𝑦^ℎ1_𝑖
        dWeights = [[learnSpeed*delta[j]*outputLayer1[i] for j in range(self.nLayer2)] for i in range(self.nLayer1)]

        # update bias weight
        dWeights.append([learnSpeed * delta[j] for j in range(self.nLayer2)])

        # 𝑤^ℎ2_𝑖𝑘 ← 𝑤^ℎ2_𝑖𝑘 + Δ𝑤^ℎ2_𝑖𝑘(𝑡) + 𝛼Δ𝑤^ℎ2_𝑖𝑘(𝑡 − 1)
        self.plus(self.weights2, dWeights, 1.0)
        self.plus(self.weights2, self.dWeights2, momentum)
        self.dWeights2 = dWeights

        # PRVNI SKRYTA VRSTVA

        sum = [0.0 for i in range(self.nLayer1)]
        for i in range(self.nLayer1):
            for j in range(self.nLayer2):
                sum[i]+= delta[j] * self.weights2[i][j]

        delta = [self.GAMMA*(outputLayer1[i]*(1-outputLayer1[i])*sum[i]) for i in range(self.nLayer1)]

        dWeights = [[learnSpeed * delta[j] * input[i] for j in range(self.nLayer1)] for i in range(self.nInput)]

        dWeights.append([learnSpeed * delta[j] for j in range(self.nLayer1)])

        self.plus(self.weights1, dWeights, 1.0)
        self.plus(self.weights1, self.dWeights1, momentum)
        self.dWeights1 = dWeights

        return statistics.mean([outputLayer3[i]-expectedOutput[i] for i in range(self.nOutput)])

    def plus(self, a, b, times):
        """
        Matrix addition, with multiplication of added values.

        Performs: A = A + (times * B)

        Args:
            a: Matrix A
            b: Matrix B
            times: Multiplier

        Returns:
            Resulting matrix
        """

        for i in range (len(a)):
            for j in range(len(a[i])):
                a[i][j] += times * b[i][j]
