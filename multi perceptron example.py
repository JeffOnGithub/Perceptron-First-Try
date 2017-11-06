# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:32:56 2017

@author: Jeff@grenier.cc

Perceptron v0.1
"""

import numpy as np
from Perceptron import Perceptron


"""
Multilayer perceptron example using sigmoid
"""

# Reference data
#x1 = np.array([0, 0, 1, 1])
#x2 = np.array([0, 1, 0, 1])
#y  = np.array([0, 0, 0, 1])

raw_x1 = [20, 16, 15, 5, 16, 2, 6, 20, 30, 0, 12, 3, 5]
raw_x1_sum = sum(raw_x1)
raw_x2 = [7, 10, 5, 15, 6, 20, 16, 11, 0, 51, 25, 50, 30]
raw_x2_sum = sum(raw_x2)

#Normalize data
x1 = np.array([float(i)/sum(raw_x1) for i in raw_x1])
x2 = np.array([float(i)/sum(raw_x2) for i in raw_x2])
y  = np.array([1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0])

#Build the hidden layer
hidden_layer = [Perceptron(1, np.array([0.2, 0.1, 0.3]), 'sigmoid', 'layer'),
                Perceptron(1, np.array([-0.3, -0.2, 0.4]), 'sigmoid', 'layer')]

#Build the output layer
output_layer = Perceptron(1, np.array([0.4, 0.5, -0.4]), 'sigmoid', 'output')


#Training
for x in range(0, 10000):
    for index in range(0, y.size):
        #Take an observation
        data_points = np.array([np.take(x1, index), np.take(x2, index)])
        
        #Feed it to the hidden layer
        hidden_layer_results = np.array([
                hidden_layer[0].activation(data_points),
                hidden_layer[1].activation(data_points)])
        
        #Feed the hidden layer results to the output layer
        predicted = output_layer.activation(hidden_layer_results)
        
        #Test the result, adjust weight on wrong answer
        expected = np.take(y, index)
        
        if predicted != expected:
            #Apply gradient
            output_layer.learn(hidden_layer_results, expected, predicted, 1, 1)
            hidden_layer[0].learn(data_points, expected, hidden_layer_results[0], output_layer.saved_gradient_factor, output_layer.weights[1])
            hidden_layer[1].learn(data_points, expected, hidden_layer_results[1], output_layer.saved_gradient_factor, output_layer.weights[2])
            
            
#Display result
print(hidden_layer[0].weights)        
print(hidden_layer[1].weights) 
print(output_layer.weights)

for index in range(0, y.size):
    x1_element = np.take(x1, index)
    x2_element = np.take(x2, index)
    y_element = np.take(y, index)

    #Feed it to the hidden layer
    hidden_layer_results = np.array([
       hidden_layer[0].activation(np.array([x1_element, x2_element])),
       hidden_layer[1].activation(np.array([x1_element, x2_element]))])
        
    #Feed the hidden layer results to the output layer
    predict_element = output_layer.activation(hidden_layer_results)
        
    print(str("{:10.0f}".format(x1_element* raw_x1_sum)) + "\t" + 
          str("{:10.0f}".format(x2_element* raw_x1_sum)) + "\t" + 
          str("{:10.0f}".format(y_element)) + "\t" + 
          str("{:10.4f}".format(predict_element)))
        

