import csv

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from network import Network
from perceptron import Perceptron


def sigmoid(phi):
    return np.round(1.0 / (1.0 + np.exp(-phi)), 15)


def graph(label_x, data_x, label_y, data_y):
    fig, axs = plt.subplots(1, 1)
    axs.plot(data_x, data_y)
    axs.set_xlabel(label_x)
    axs.set_ylabel(label_y)
    axs.grid(True)
    plt.show()


def normalize(value, minimum, maximum):
    return (value - minimum) / (maximum - minimum)


def parse_image(url, width, height):
    image = Image.open(url)
    image = image.convert('L')
    image = image.resize((width, height), Image.ANTIALIAS)
    image = np.array(image, dtype=np.float)
    return image


def get_image_window(image, column_param, height_param, size):
    result = np.zeros((size, size))
    c = 0
    r = 0

    image_array = parse_image(image[0], image[1], image[2])

    for row in range(height_param, height_param + size):
        r = 0
        for column in range(column_param, column_param + size):
            result[c][r] = image_array[row][column]
            r += 1
        c += 1

    return np.array(result)


def task_xor():
    net.initialize_network([Perceptron(2, 2, sigmoid), Perceptron(2, 1, sigmoid)], -0.3, 0.3)

    # Data load
    training_set = np.array([
        np.array([np.array([0, 0]), np.array([0])]),
        np.array([np.array([1, 0]), np.array([1])]),
        np.array([np.array([0, 1]), np.array([1])]),
        np.array([np.array([1, 1]), np.array([0])]),
    ])
    print("Before learn 0,0:", net.feed_forward(training_set[0][0]))
    # Graph helper
    epoch_array = []
    error_array = []
    average_error = 0
    total_error = 0
    learning_speed = 0.8
    inertia = 0.5
    epochs = 10000
    for i in range(epochs):

        average_error = 0
        net.epoch_start()
        for part in training_set:
            average_error += net.learn(part[0], part[1])
        net.epoch_finish(learning_speed, inertia)
        total_error = average_error / len(training_set)
        # net.print_net()

        if total_error < 0.05:
            print("EPOCH: {:d}".format(i + 1))
            print("ERROR: {:f}".format(total_error))
            print("========================")
            break

        error_array.append(total_error)
        epoch_array.append(i + 1)

        if i % 100 == 0:
            print("EPOCH: {:d}".format(i + 1))
            print("ERROR: {:f}".format(total_error))
            print("========================")
    # net.print_net()
    graph("Epoch", epoch_array, "Error", error_array)
    print("After learn 0,0:", net.feed_forward(training_set[0][0]))
    print("After learn 0,1:", net.feed_forward(training_set[1][0]))
    print("After learn 1,0:", net.feed_forward(training_set[2][0]))
    print("After learn 1,1:", net.feed_forward(training_set[3][0]))
    # net.debug_log()


def task_iris():
    net.initialize_network([Perceptron(4, 4, sigmoid), Perceptron(4, 3, sigmoid)], -0.3, 0.3)

    # Data load
    filename = 'iris.data'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    temp_list = list(reader)

    training_set = []
    for record in temp_list:
        if len(record) > 0:
            training_set.append(
                [
                    np.array([
                        normalize(float(record[0]), 4.3, 7.9),
                        normalize(float(record[1]), 2, 4.4),
                        normalize(float(record[2]), 1, 6.9),
                        normalize(float(record[3]), 0.1, 2.5)
                    ]),
                    np.array([
                        1 if record[4] == "Iris-virginica" else 0,
                        1 if record[4] == "Iris-versicolor" else 0,
                        1 if record[4] == "Iris-setosa" else 0
                    ])
                ]
            )

    # Graph helper
    epoch_array = []
    error_array = []
    average_error = 0
    total_error = 0
    learning_speed = 0.05
    inertia = 0.3
    epochs = 10000
    for i in range(epochs):

        average_error = 0
        net.epoch_start()
        for part in training_set:
            average_error += net.learn(part[0], part[1])
        net.epoch_finish(learning_speed, inertia)
        total_error = average_error / len(training_set)

        if total_error < 0.009:
            print("EPOCH: {:d}".format(i + 1))
            print("ERROR: {:f}".format(total_error))
            print("========================")
            break

        error_array.append(total_error)
        epoch_array.append(i + 1)

        if i % 100 == 0:
            print("EPOCH: {:d}".format(i + 1))
            print("ERROR: {:f}".format(total_error))
            print("========================")

    graph("Epoch", epoch_array, "Error", error_array)

    print(" ")
    print("CODING")
    print("------------------------")
    print("Virginica:  [1 0 0]")
    print("Versicolor: [0 1 0]")
    print("Setosa:     [0 0 1]")
    print("------------------------")
    print("After learn " + str(training_set[0][1]) + " :", np.round(net.feed_forward(training_set[0][0])))
    print("After learn " + str(training_set[55][1]) + " :", np.round(net.feed_forward(training_set[55][0])))
    print("After learn " + str(training_set[107][1]) + " :", np.round(net.feed_forward(training_set[107][0])))
    print("After learn " + str(training_set[142][1]) + " :", np.round(net.feed_forward(training_set[142][0])))
    print("After learn " + str(training_set[9][1]) + " :", np.round(net.feed_forward(training_set[9][0])))


def task_draw():
    net.initialize_network([Perceptron(25, 25, sigmoid), Perceptron(25, 20, sigmoid)], -0.5, 0.5)

    # Data load
    filename = 'imgs.data'
    raw_data = open(filename, 'rt')
    reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
    temp_list = list(reader)

    training_set = []

    for record in temp_list:
        if len(record) > 0:
            label = list(record[1])
            label = np.array(label, dtype=np.float)

            image = parse_image("./imgs/" + record[0], 5, 5)
            image = np.concatenate(image)
            image = normalize(image, 0, 255)

            training_set.append(
                [
                    image,
                    label,
                    record[0]
                ]
            )

    # Graph helper
    epoch_array = []
    error_array = []

    average_error = 0
    total_error = 0
    learning_speed = 0.05
    inertia = 0.5
    epochs = 5000

    for i in range(epochs):
        average_error = 0
        net.epoch_start()
        for part in training_set:
            average_error += net.learn(part[0], part[1])
        net.epoch_finish(learning_speed, inertia)
        total_error = average_error / len(training_set)

        if total_error < 0.00005:
            print("EPOCH: {:d}".format(i + 1))
            print("ERROR: {:f}".format(total_error))
            print("========================")
            break

        error_array.append(total_error)
        epoch_array.append(i + 1)

        if i % 50000 == 0:
            error_array = []
            epoch_array = []

        if i % 1000 == 0:
            print("EPOCH: {:d}".format(i + 1))
            print("ERROR: {:f}".format(total_error))
            print("========================")

    graph("Epoch", epoch_array, "Error", error_array)

    for i in range(20):
        print("Train set      :" + str(training_set[i][1]) + " :")
        print("After learn    :" + str(np.round(net.feed_forward(training_set[i][0]))))
        print("========================")

    width = 90
    height = 105

    # new image file
    result_image = Image.new('RGB', (width, height), (255, 255, 255))

    scan_window = 5

    print("Drawing Image")
    for row in range(0, (height - (scan_window - 1)), 1):
        for column in range(0, (width - (scan_window - 1)), 1):

            # get image
            img = get_image_window(["./imgs/90x105.png", 90, 105], column, row, scan_window)
            img2 = np.round(np.concatenate(img))
            img2 = normalize(img2, 0, 255)

            # get net result
            net_result = net.feed_forward(img2)
            net_max = max(net_result)
            result = [i for i, j in enumerate(net_result) if j == net_max]

            if len(result) > 0:

                if result[0] != 0:
                    result_image.paste(Image.open("./imgs/" + str(result[0]) + ".png"), (column, row))
    result_image.save('./imgs/out.png')

    print("Opening Image")
    imgshow = Image.open('./imgs/out.png')
    imgshow.show()


if __name__ == "__main__":
    net = Network()

    # task_xor()
    # task_iris()
    task_draw()
