# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 10:32:56 2017

@author: Jeff@grenier.cc

Perceptron v0.1
"""

import numpy as np
import math

"""
Perceptron class
"""
class Perceptron:

    learning_step = 0
    weights = np.array([])
    activation_function = 0
    position = 0
    saved_gradient_factor = 1
    
    #Init the perceptron
    def __init__(self, learning_step, weights, activation_function, position):
        self.learning_step = learning_step
        self.weights = weights
        
        if activation_function == 'sigmoid':
            self.activation_function = 0
        elif activation_function == 'heaviside':
            self.activation_function = 1
        else:
            raise NameError("Not a correct type of activation function")
        
        if position == 'single':
            self.position = 0
        elif position == 'layer':
            self.position = 1
        elif position == 'output':
            self.position = 2
        else:
            raise NameError("Not a correct type of perceptron position")
        
    def activation(self, data_points):
        if self.activation_function == 0:
            return self.activation_sigmoid(self.weights, data_points)
        if self.activation_function == 1:
            return self.activation_heaviside(self.weights, data_points) 
    
    #Activation function Heaviside
    #threshold must always be first element in weights
    #data_points length = weights length - 1
    def activation_heaviside(self, weights, data_points):
        result = 0
        data_point = np.nditer(data_points, flags=['f_index'])
        #Sum of factors by weights
        while not data_point.finished:
            #print(data_point[0])
            result += data_point[0] * np.take(weights, data_point.index + 1)
            data_point.iternext()
        #Add the threshold and check if we are over
        #print(result)
        if result + weights[0] >= 0:
            return 1
        else:
            return 0
    
    #Activation function sigmoid
    #threshold must always be first element in weights
    #data_points length = weights length - 1
    def activation_sigmoid(self, weights, data_points):
        result = 0
        data_point = np.nditer(data_points, flags=['f_index'])
        #Sum of factors by weights
        while not data_point.finished:
            #print(data_point[0])
            result += data_point[0] * np.take(weights, data_point.index + 1)
            data_point.iternext()
        #Add the threshold
        result += weights[0]
        #Return the sigmoid form
        #Prevent overflows
        if abs(result) < 0.0002:
            return 0.5
        elif abs(result) > 500:
            return 0
        else:
            return 1 / (1 + math.pow(2.71828182846, -1 * result))
    
    def learn(self, data_points, expected, predicted, forward_gradient_factor, forward_weight):
        if self.position == 0:
            self.learn_single(data_points, expected, predicted, forward_gradient_factor, forward_weight)
        if self.position == 1:
            self.learn_layer(data_points, expected, predicted, forward_gradient_factor, forward_weight)
        if self.position == 2:
            self.learn_output(data_points, expected, predicted, forward_gradient_factor, forward_weight)    
        
    def learn_single(self, data_points, expected, predicted, forward_gradient_factor, forward_weight):
        new_weights = []
        
        value = np.nditer(self.weights, flags=['f_index'])
        while not value.finished:
            if value.index == 0:
                from_factor = 1
            else:
                from_factor = np.take(data_points, value.index - 1)
            new_weights.append(value[0] + self.learning_step * (expected - predicted) *  from_factor)
            value.iternext()
        self.weights = np.array(new_weights)        

    def learn_layer(self, data_points, expected, predicted, forward_gradient_factor, forward_weight):
        #Add the threshold constant
        data_points = np.insert(data_points, 0 ,1)
        #Calculate gradient factor
        gradient_factor = predicted * (1 - predicted) * forward_gradient_factor * forward_weight
        #print(predicted)
        #print(forward_gradient_factor)
        #print(forward_weight)
        #Calculate delta for all weights
        new_weights = []
        value = np.nditer(self.weights, flags=['f_index'])
        while not value.finished:
            data_point = np.take(data_points, value.index)
            #print(self.learning_step)
            #print(gradient_factor)
            #print(data_point)
            new_weights.append((self.learning_step * gradient_factor * data_point) + value[0])
            value.iternext()
        #Apply the delta to get the new weights
        self.weights = np.array(new_weights)  

    def learn_output(self, data_points, expected, predicted, forward_gradient_factor, forward_weight):
        #Add the threshold constant
        data_points = np.insert(data_points, 0 ,1)
        #Calculate gradient factor
        gradient_factor = (expected - predicted) * predicted * (1 - predicted)
        #Save gradient factor
        self.saved_gradient_factor = gradient_factor
        #Calculate delta for all weights
        new_weights = []
        value = np.nditer(self.weights, flags=['f_index'])
        while not value.finished:
            data_point = np.take(data_points, value.index)
            new_weights.append((self.learning_step * gradient_factor * data_point) + value[0])
           
            value.iternext()
        #Apply the delta to get the new weights
        self.weights = np.array(new_weights)  
