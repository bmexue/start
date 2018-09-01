from numpy import exp,array,random,dot
import sys
from numpy import *
import numpy as np
import operator
import csv
import math
import time

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2*random.random((3,1))-1

    def __sigmoid(self,x):
        return 1/(1+exp(-x))


    def  __sigmoid_derivative(self,x):
        return x*(1-x)
    # 一次学习m个
    def train(self,training_set_inputs,training_set_outputs,number):
        for i in xrange(number):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = dot(training_set_inputs.T,error*self.__sigmoid_derivative(output))
            self.synaptic_weights +=adjustment
            print "error:"
            print error
            

    def think(self,inputs):
        return self.__sigmoid(dot(inputs,self.synaptic_weights))

if __name__ == '__main__':
    neural_network = NeuralNetwork()
    print "Init synaptic_weights :\n"
    print neural_network.synaptic_weights
    training_set_inputs = array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    training_set_outputs = array([[0,1,0,1,0,1,1,1]]).T
    neural_network.train(training_set_inputs,training_set_outputs,1000)
    print "Last synaptic_weights :\n"
    print neural_network.synaptic_weights
    print "think 1,0,0: \n"
    print neural_network.think(array([1,0,0]))

