import random
import statistics
import sys

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

# convert image to 1D array (scanning left to right) of 0s and 1s
from ThreeLayerNeuralNetwork import ThreeLayerNeuralNetwork


def imageToArray(src):
    img = Image.open(src).convert('L')
    np_img = np.matrix(img)
    np_img = ~np_img
    np_img[np_img > 0] = 1
    return np.asarray(np_img).reshape(-1).tolist()


def verifyNetwork(printStats = False):

    successCount = 0

    if (printStats):
        print("\n\nIMG\tDET\tOK\tMEAN ERROR\tRAW RESPONSE\n---------------------------------------")

    for i in range(NUMBER_OF_IMAGES):
        response = network.recall(trainingImages[i])
        detectedImageType = response.index(max(response))

        success = (detectedImageType == i)
        meanError = statistics.mean([response[x] - correctResults[i][x] for x in range(NUMBER_OF_IMAGES)])

        if (success):
            successCount += 1

        if (printStats):
            print("%d\t%d\t%s\t%f\t%s" % (i, detectedImageType, "✔" if success else "❎", meanError, response))

    if (printStats):
        print("---------------------------------------\n")

    return successCount


if __name__ == "__main__":

    NUMBER_OF_IMAGES = 20
    IMAGE_RESOLUTION = 5 * 5

    #load images (index = image type)
    trainingImages = [imageToArray("images/"+str(i).rjust(2, '0')+".png") for i in range(NUMBER_OF_IMAGES)]

    # correct results (index = image type)
    correctResults = np.asarray(np.identity(NUMBER_OF_IMAGES)).tolist()

    # initialize network
    network = ThreeLayerNeuralNetwork(IMAGE_RESOLUTION, 25, 25, NUMBER_OF_IMAGES)

    #training

    LEARN_SPEED = 0.2
    MOMENTUM = 0.8

    network.randomizeWeights()

    meanErrorInEpochs = []
    successCountInEpochs = []

    for epoch in range(1000):

        #epoch


        randomizedImages = [i for i in range(NUMBER_OF_IMAGES)]
        random.shuffle(randomizedImages)

        errorSum = 0;

        for imageUnderTest in randomizedImages:
            errorSum += network.learn(trainingImages[imageUnderTest], correctResults[imageUnderTest], LEARN_SPEED, MOMENTUM)

        epochMeanError = errorSum / len(randomizedImages)
        meanErrorInEpochs.append(epochMeanError)



        # if epoch % 10 == 0:
        sys.stdout.write("\rEpoch %d, error:\t%f" % (epoch, epochMeanError))
        sys.stdout.flush()

        successCountInEpochs.append(verifyNetwork(epoch % 100 == 0))

    verifyNetwork()

    fig, host = plt.subplots()
    par1 = host.twinx()

    p1, = host.plot(meanErrorInEpochs, "r-", label="Mean Error")
    p2, = par1.plot(successCountInEpochs, "g-", label="Success count")

    host.set_xlabel("Epoch number")
    host.set_ylabel("Mean error")
    par1.set_ylabel("Number of successfully recognized images")

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    par1.set_yticks(np.arange(0, NUMBER_OF_IMAGES+1, 1.0), minor=False)
    host.grid()

    plt.show()

    pass;


