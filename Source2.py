from keras.datasets import mnist
import matplotlib.pyplot as plt
import scipy.special
import numpy as np
from random import *
# n=6
(x_train, y_train) , (x_test, y_test) = mnist.load_data()
# print(y_train[n])
# plt.imshow((255-x_train[n])/255, cmap="gray")
# plt.show()

class NeuralNetwork:
    def __init__(self, input, rate):
        self.i = input
        self.r = rate
        self.activation = lambda x : scipy.special.expit(x)
        self.how = np.random.random((1, 1))
        self.ihw = np.random.random((self.i, 1))

    def Itrain(self, input_list, output_list):
        buf = np.array(input_list/255)
        buf.shape = (784, 1)
        #print(buf.shape)
        hidden_result = self.activation(np.dot(buf.transpose(), self.ihw))
        hidden_error = output_list - hidden_result
        #print(error.shape)
        self.ihw += self.r * np.dot(buf, hidden_result * (1 - hidden_result) * hidden_error)
        result = self.activation(np.dot(hidden_result, self.how))
        error = output_list - result
        self.how += self.r * np.dot(hidden_result, result * (1 - result) * error)

    def think(self, testinput, testoutput):
        print(testoutput)
        testinput = testinput/255
        testinput.shape = (784, 1)
        return np.dot(self.how.transpose(), testinput)

    def set_r(self, new_rate):
        self.r = new_rate


if __name__ == "__main__":
    NN = NeuralNetwork(784, 0.1)
    for iteration in range(1, 59999):
        NN.Itrain(x_train[iteration], y_train[iteration])
    plt.imshow((255 - x_test[2])/255, cmap='gray' )
    plt.show()
    test = NN.think(x_test[2], y_test[2])
    print(test)





    # def training(self):
    #     inputlist = np.array(x_train)
    #     outputlist = np.array(y_train)
    #     out_results = self.activation(np.dot(self.iow, inputlist))
    #     self.iow += self. r * np.dot(out_results-outputlist, inputlist * out_results *(out_results - 1) )
    #
    # def target(self, targetinput, targetresults):
    #     for n in range(1, )
    #     return self.activation(np.dot(self.iow, targetinput))













