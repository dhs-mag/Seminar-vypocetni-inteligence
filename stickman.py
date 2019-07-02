import colorsys
import math
import random
import statistics
import sys

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

from ThreeLayerNeuralNetwork import ThreeLayerNeuralNetwork
from TwoLayerNeuralNetwork import TwoLayerNeuralNetwork


def imageToArray(src):
    """
    Convert image to 1D array (scanning left to right) of 0s and 1s
    """
    return np.asarray(imageToMatrix(src)).reshape(-1).tolist()


def imageToMatrix(src):
    """
    Convert image to 2D matrix of 0s and 1s
    """
    img = Image.open(src).convert('L')
    np_img = np.matrix(img)
    np_img = ~np_img
    np_img = np_img/255.0
    # np_img[np_img > 0] = 1
    return np_img


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

def convertColors(sourceImg, imageType):
    datas = sourceImg.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:

            # newData.append((0, 0, 0, 100))

            rgb = colorsys.hsv_to_rgb((1.0 / NUMBER_OF_IMAGES) * imageType, 1.0, 0.5)
            newData.append((
                math.floor(rgb[0] * 255),
                math.floor(rgb[1] * 255),
                math.floor(rgb[2] * 255),
                127))

    sourceImg.putdata(newData)
    return sourceImg


if __name__ == "__main__":

    NUMBER_OF_IMAGES = 20
    SAMPLE_SIZE = 5
    IMAGE_RESOLUTION = SAMPLE_SIZE * SAMPLE_SIZE

    #load images (index = image type)
    trainingImages = [imageToArray("images/"+str(i).rjust(2, '0')+".png") for i in range(NUMBER_OF_IMAGES)]

    # correct results (index = image type)
    correctResults = np.asarray(np.identity(NUMBER_OF_IMAGES)).tolist()

    # initialize network
    # network = ThreeLayerNeuralNetwork(IMAGE_RESOLUTION, 25, 25, NUMBER_OF_IMAGES)
    network = TwoLayerNeuralNetwork(IMAGE_RESOLUTION, 25, NUMBER_OF_IMAGES)

    # TRAINING

    LEARN_SPEED = 0.2
    MOMENTUM = 0.8

    network.randomizeWeights()

    meanErrorInEpochs = []
    successCountInEpochs = []


    for epoch in range(500):

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

    # plot training results
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

    # STICKMAN PROCESSING

    print("\n\nProcessing stickman\n-------------------------")

    stickmanData = imageToMatrix("images/test/stickman.png")

    results = np.zeros((stickmanData.shape[0] - SAMPLE_SIZE, stickmanData.shape[1] - SAMPLE_SIZE))

    resultImage = Image.new('RGB', (stickmanData.shape[1], stickmanData.shape[0]), (255, 255, 255))

    for y in range(stickmanData.shape[0] - SAMPLE_SIZE):
        for x in range(stickmanData.shape[1] - SAMPLE_SIZE):

            sampleSubmatrix = stickmanData[y:y + SAMPLE_SIZE, x:x + SAMPLE_SIZE]
            sampleList = np.asarray(sampleSubmatrix).reshape(-1).tolist()

            #recall in neural network
            recall = network.recall(sampleList)
            detectedImageType = recall.index(max(recall))
            results[y,x] = detectedImageType

            # process result image
            resultPatch = Image.open("images/" + str(detectedImageType).rjust(2, '0') + ".png").convert("RGBA")
            resultPatch = convertColors(resultPatch, detectedImageType)

            resultImage.paste(resultPatch, (x, y), resultPatch)  # third arg to achieve transparency

            sys.stdout.write("\rProcessing Y: %d\t X: %d\t => Detected:\t%d" % (y, x, detectedImageType))
            sys.stdout.flush()

    # save and plot image

    resultImage.save("result.png","PNG")

    dpi = 80.0
    xpixels, ypixels = stickmanData.shape[0], stickmanData.shape[1]

    fig = plt.figure(figsize=(ypixels / dpi, xpixels / dpi), dpi=dpi)
    fig.figimage(resultImage)
    plt.show()



