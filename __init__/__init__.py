# def main():
#     net = Net()
#
#     net.add_neuron(Perceptron(4, 4, sigmoid))
#     net.add_neuron(Perceptron(3, 4, sigmoid))
#
#     net.net_init(-0.3, 0.3)
#
#     filename = 'iris.csv'
#     raw_data = open(filename, 'rt')
#     reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
#     x = list(reader)
#
#     train_set = []
#
#     for item in x:
#         if len(item) > 0:
#             train_set.append(
#                 [
#                     np.array([
#                         normalize(float(item[0]), 4.3, 7.9),
#                         normalize(float(item[1]), 2, 4.4),
#                         normalize(float(item[2]), 1, 6.9),
#                         normalize(float(item[3]), 0.1, 2.5)
#                     ]),
#                     np.array([
#                         1 if item[4] == "Iris-virginica" else 0,
#                         1 if item[4] == "Iris-versicolor" else 0,
#                         1 if item[4] == "Iris-setosa" else 0
#                     ])
#                 ])
#
#     errors = []
#     epoch = []
#
#     for i in range(10000):
#         avg_error = 0
#         net.epoch_start()
#         for pat in train_set:
#             avg_error += net.learn(pat[0], pat[1])
#         net.epoch_finish()
#         error = avg_error / len(train_set)
#         errors.append(error)
#         epoch.append(i + 1)
#
#         if i % 100 == 0:
#             print("EPOCH:", i + 1)
#             print("Error:", error)
#             print("----------------------------")
#             if error < 0.009:
#                 break
#
#     # graph
#     figure, axis = plot.subplots(1, 1)
#     axis.plot(epoch, errors)
#     axis.set_xlabel('Epoch')
#     axis.set_ylabel('Error')
#     axis.grid(True)
#     plot.show()
#
#     print("After learn " + str(train_set[0][1]) + " :", np.round(net.recall(train_set[0][0])))
#     print("After learn " + str(train_set[55][1]) + " :", np.round(net.recall(train_set[55][0])))
#     print("After learn " + str(train_set[107][1]) + " :", np.round(net.recall(train_set[107][0])))
#     print("After learn " + str(train_set[142][1]) + " :", np.round(net.recall(train_set[142][0])))
#
#
# if __name__ == "__main__":
#     main()