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

    # trainSet = np.array([
    #     np.array([np.array([0, 0]), np.array([0])]),
    #     np.array([np.array([1, 0]), np.array([1])]),
    #     np.array([np.array([0, 1]), np.array([1])]),
    #     np.array([np.array([1, 1]), np.array([0])]),
    # ])

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

    net.print_net(outfile)

    avgErr = 0
    err = 0
    for i in range(24):
        print("# epocha", i+1, file=outfile)

        avgErr = 0
        net.epochStart()
        iteration = 1
        for pat in trainSet:
            print("# iterace", iteration, file=outfile)
            avgErr += net.learn(pat[0], pat[1])

            print(("  % 1.15f" % pat[0][0]).replace('.', ',') + " ; input [0]", file=outfile)
            print(("  % 1.15f" % pat[0][1]).replace('.', ',') + " ; input [1]", file=outfile)
            net.print_net(outfile)

            iteration += 1
        net.epochFinish()
        err = avgErr/len(trainSet)

        print("# weight update", file=outfile)
        print(("  % 1.15f" % err).replace('.', ',') + " ; mse", file=outfile)
        net.print_net(outfile)

        if err < 0.05:
            break
        # print("Error:", err)
        # print("========================")

    net.print_net()

    print("After learn 0,0:", net.recall(trainSet[0][0]))
    print("After learn 1,0:", net.recall(trainSet[1][0]))
    print("After learn 0,1:", net.recall(trainSet[2][0]))
    print("After learn 1,1:", net.recall(trainSet[3][0]))
