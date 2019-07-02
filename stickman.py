import random
import statistics

import numpy as np
from PIL import Image

# convert image to 1D array (scanning left to right) of 0s and 1s
from NeuralNetwork import NeuralNetwork


def imageToArray(src):
    img = Image.open(src).convert('L')
    np_img = np.matrix(img)
    np_img = ~np_img
    np_img[np_img > 0] = 1
    return np.asarray(np_img).reshape(-1).tolist()


def verifyNetwork():
    print("IMG\tDET\tSUCCESS\tMEAN ERROR\tRAW RESPONSE\n---------------")
    for i in range(NUMBER_OF_IMAGES):
        response = network.classifyOut(trainingImages[i])
        detectedImageType = response.index(max(response))

        success = (detectedImageType == i)
        meanError = statistics.mean([response[x] - correctResults[i][x] for x in range(NUMBER_OF_IMAGES)])

        print("%d\t%d\t%s\t%f\t%s" % (i, detectedImageType, success, meanError, response))


if __name__ == "__main__":

    NUMBER_OF_IMAGES = 20
    IMAGE_RESOLUTION = 5 * 5

    #load images (index = image type)
    trainingImages = [imageToArray("images/"+str(i).rjust(2, '0')+".png") for i in range(NUMBER_OF_IMAGES)]

    # correct results (index = image type)
    correctResults = np.asarray(np.identity(NUMBER_OF_IMAGES)).tolist()

    # initialize network
    network = NeuralNetwork(IMAGE_RESOLUTION, 25, 25, NUMBER_OF_IMAGES)

    #training

    LEARN_SPEED = 0.2
    MOMENTUM = 0.8

    network.randomizeWeights()


    for epoch in range(1000):

        #epoch

        print("Epoch %d, error:\t" % (epoch), end="")

        randomizedImages = [i for i in range(NUMBER_OF_IMAGES)]
        random.shuffle(randomizedImages)

        errorSum = 0;

        for imageUnderTest in randomizedImages:
            errorSum += network.learnBPROP(trainingImages[imageUnderTest], correctResults[imageUnderTest], LEARN_SPEED, MOMENTUM)

        epochMeanError = errorSum / len(randomizedImages)
        print(epochMeanError)

        if epoch % 100 == 0:
            verifyNetwork()


    pass;


