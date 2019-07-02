from PIL import Image
import numpy as np

from Net import Net


# https: // stackoverflow.com / questions / 50494541 / bw - image - to - binary - array
def loadImageToArray(src):

    img = Image.open(src).convert('L')

    np_img = np.matrix(img).flatten()
    np_img = np.squeeze(np.asarray(np_img))
    np_img = ~np_img  # invert B&W
    np_img[np_img > 0] = 1

    return np_img


if __name__ == "__main__":

    trainSet = []

    for patternId in range(0, 20):

        loadedImage = loadImageToArray("images/"+str(patternId).rjust(2, '0')+".png")
        correctResult = np.zeros(20)
        correctResult[patternId] = 1

        element = np.array([
            loadedImage,
            correctResult
        ])

        trainSet.append(element)

    net = Net()
    net.netInit(-0.3, 0.3)


    print(trainSet[0][0])

    print("Before learn:", net.recall(trainSet[0][0]))

    outfile = open('output/code_images.txt', 'w')

    avgErr = 0
    err = 0
    for i in range(1000):
        print("# epocha", i+1, file=outfile)

        avgErr = 0
        iteration = 1
        for pat in trainSet:
            print("# iterace", iteration, file=outfile)
            avgErr += net.learn(pat[0], pat[1])

            print(("  % 1.15f" % pat[0][0]).replace('.', ',') + " ; input [0]", file=outfile)
            print(("  % 1.15f" % pat[0][1]).replace('.', ',') + " ; input [1]", file=outfile)

            iteration += 1
        err = avgErr/len(trainSet)

        print("# weight update", file=outfile)
        print(("  % 1.15f" % err).replace('.', ',') + " ; mse", file=outfile)

        # if err < 0.05:
        #     break
        print("Error:", err)
        # print("========================")

    for patternId in range(0, 20):
        print("After learn "+str(patternId)+": ")

        output = net.recall(trainSet[0][0])

        print("[", end='')
        for x in output:
            print(("%1.15f, " % x), end='')
        print("]")
