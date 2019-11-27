from numpy import exp, array, random, dot
import scipy as sc
import math as m
import matplotlib.pyplot as plt

training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_set_outputs = array([[0, 1, 1, 0]]).T
random.seed(1)
synaptic_weights = 2 * random.random((3, 1))
test = []
x = [i for i in range (100000)]
for iteration in range(100000):
    output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
    #print(output)
    synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
    test.append(output)
print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))

from PIL import Image

# im = Image.open('dead_parrot.jpg') # Can be many different formats.
# pix = im.load()
# print im.size  # Get the width and hight of the image for iterating over
# print pix[x,y]  # Get the RGBA Value of the a pixel of an image
# pix[x,y] = value  # Set the RGBA Value of the image (tuple)
# im.save('alive_parrot.png')  # Save the modified pixels as .png

