# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:32:56 2017

@author: Jeff@grenier.cc

Perceptron v0.1
"""

import numpy as np
from Perceptron import Perceptron


"""
Single perceptron example using heaviside
Typical linear classifier
"""

# Reference data
x1 = np.array([20, 20, 16, 15, 5, 16, 2, 6, 20, 30, 0, 12, 3, 5])
x2 = np.array([7, 7, 10, 5, 15, 6, 20, 16, 11, 0, 51, 25, 50, 30])
y  = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

#Build a perceptron
perceptron = Perceptron(0.1, np.array([0, 0, 0]), 'heaviside', 'single')

#Training
for x in range(0,5000):
    for index in range(0, y.size - 1):
        #Take an observation
        data_points = np.array([np.take(x1, index), np.take(x2, index)])
        #Run it in activation
        predicted = perceptron.activation(data_points)
        expected = np.take(y, index)
        #Test the result, adjust weight on wrong answer
        if predicted != expected:
            perceptron.learn(data_points, expected, predicted, 1, 1)

            
#Display result
print(perceptron.weights)

for index in range(0, y.size - 1):
    x1_element = np.take(x1, index)
    x2_element = np.take(x2, index)
    y_element = np.take(y, index)
    predict_element = perceptron.activation(np.array([x1_element, x2_element]))
    print(str(x1_element) + "\t" + str(x2_element) + "\t" + str(y_element) + "\t" + str(predict_element))
        

