
import random
import math

#
#Universal MLP feed-forward neural network with two hidden layers.
#
#
class NeuralNetwork:

     #  nLayer1, nLayer2
     #  nInput, nOutput
     # double[][] weights1, weights2, weights3
     # double[][] dWeights1, dWeights2, dWeights3

    #@param nLayer1 Number of neurons in first hidden layer
    #@param nLayer2 Number of neurons in second hidden layer
    #@param nInput Number of inputs - how many image features are there
    #@param nOutput Number of outputs - how many image classes we want to detect
    def __init__(self, nLayer1,  nLayer2,  nInput,  nOutput):
        self.GAMMA = 1.0

        self.nLayer1 = nLayer1
        self.nLayer2 = nLayer2
        self.nOutput = nOutput
        self.nInput = nInput
        self.weights1 = [[0 for i in range(nLayer1)] for j in range(nInput+1)]
        self.weights2 = [[0 for i in range(nLayer2)] for j in range(nLayer1+1)]
        self.weights3 = [[0 for i in range(nOutput)] for j in range(nLayer2+1)]
        self.dWeights1 = [[0 for i in range(nLayer1)] for j in range(nInput+1)]
        self.dWeights2 = [[0 for i in range(nLayer2)] for j in range(nLayer1+1)]
        self.dWeights3 = [[0 for i in range(nOutput)] for j in range(nLayer2+1)]

    #Return current weights of the neural network.
    #Usefull for storing state.
    def exportWeights(self):
        return (self.weights1, self.weights2, self.weights3)

    #
    #Set weights of the neural network.#
    #Usefull for restoring state.
    #
    #@param weights
    #
    def importWeights(self, weights):
        self.weights1 = weights[0]
        self.weights2 = weights[1]
        self.weights3 = weights[2]

    #
    #Randomize weights, use before training.
    #
    def randomizeWeights(self):
        self.randomizeWeightsMatrix(self.weights1)
        self.randomizeWeightsMatrix(self.weights2)
        self.randomizeWeightsMatrix(self.weights3)

    #
    #Randomize weights matrix.
    #
    #Individual elements will be random valued from erval <-1,1>
    #
    #@param weights Matrix to be randomized.
    #
    def randomizeWeightsMatrix(self, weights):

        for i in range(len(weights)):
            for j in range(len(weights[i])):
                weights[i][j] = 1.0 - 2.0 * random.uniform(0, 1)

    #
    #Performs classsification - data transit throughout the neural network.
    #
    #@param input Input data vector
    #@return Output vector
    #
    def classifyOut(self, input):
        outputLayer1 = [0 for i in range(self.nLayer1)]
        outputLayer2 = [0 for i in range(self.nLayer2)]
        return self.classify(input, outputLayer1, outputLayer2)

    #
    #Performs classsification - data transit throughout the neural network.
    #
    #@param input Input data vector
    #@param outputLayer1 Vector of first hidden layer
    #@param outputLayer2 Vector of second hidden layer
    #@return Output vector
    #
    def classify(self, input, outputLayer1, outputLayer2):
        output = [0 for i in range(self.nOutput)]
        self.processLayer(input, outputLayer1, self.weights1)
        self.processLayer(outputLayer1, outputLayer2, self.weights2)
        self.processLayer(outputLayer2, output, self.weights3)
        return output

    #
    #Process data tranist from one neural layer to another.
    #
    #@param input Input layer data vector.
    #@param outputLayer Output layer data vector.
    #@param weights Weights between layer vectors, weights matrix.
    #
    def processLayer(self, input, outputLayer, weights):
        for i in range(len(outputLayer)):
            outputLayer[i] = weights[input.length][i]

            for j in range(len(input)):
                outputLayer[i]+= input[j]* weights[j][i]

            outputLayer[i]*= -self.GAMMA;
            outputLayer[i] = 1.0/(1.0 + math.exp(outputLayer[i]))

    #
    #Train neural network using backpropagation method.
    #
    #@param inputData Array of input data vectors
    #@param outputData Array of expected (desired) outupt data vectors
    #@param learnSpeed Learning speeed, optimal is about 0.2
    #@param momentum Learning momentum, optimal is about 0.8
    #
    def learnBPROP(self, inputData, outputData, learnSpeed, momentum):
        for iData in range(len(inputData)):
            input = inputData[iData]
            expectedOutput = outputData[iData]
            outputLayer1 = [0 for i in range(self.nLayer1)]
            outputLayer2 = [0 for i in range(self.nLayer2)]

            # spocitame aktualni vystup neuronove site pro trenovaci tada
            outputLayer3 = self.classify(input, outputLayer1, outputLayer2)

            # VYSTUPNI VRSTVA
            delta = [self.GAMMA * (outputLayer3[i]*(1-outputLayer3[i])*(expectedOutput[i]-outputLayer3[i])) for i in range(self.nOutput)]

            dWeights = [[learnSpeed*delta[j]*outputLayer2[i] for i in range(self.nOutput)] for j in range(self.nLayer2+1)]

            for j in range(self.nOutput):
                dWeights[self.nLayer2][j] = learnSpeed*delta[j]

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
            dWeights = [[learnSpeed*delta[j]*outputLayer1[i] for i in range(self.nLayer2)] for j in range(self.nLayer1 + 1)]

            # ??
            for j in range(self.nLayer2):
                dWeights[self.nLayer1][j] = learnSpeed*delta[j]

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

            dWeights = [[learnSpeed * delta[j] * input[i] for i in range(self.nLayer1)] for j in range(self.nInput + 1)]

            for j in range(self.nLayer1):
                dWeights[self.nInput][j] = learnSpeed * delta[j]

            self.plus(self.weights1, dWeights, 1.0)
            self.plus(self.weights1, self.dWeights1, momentum)
            self.dWeights1 = dWeights

    #
    #Matrix addition, with multiplicator of added values.
    #
    #Performs: A = A + (times * B)
    #
    #@param a Matrix A
    #@param b Matrix B
    #@param times Multiplicator
    #
    def plus(self, a, b, times):
       for i in range (len(a)):
           for j in range(len(a[i])):
               a[i][j]+= times * b[i][j]
