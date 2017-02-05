import random
import os
import sys
import numpy as np
import mnist_loader

def sigmoid(x): #returns the value of sigmoid function for x
	return 1.0/(1.0+np.exp(-x))

def sigmoid_slope(x): #returns the slope of the sigmoid function for x
	return sigmoid(x)*(1-sigmoid(x))

class Neural(object):
	def __init__(self,sizes): #initialize the neural network
		self.num_layers = len(sizes) #number of layers
		self.sizes = sizes #number of neurons in respective layers 
		self.biases = [np.random.randn(i,1) for i in sizes[1:]] #randomly creating weights for each layer. randn seems to give a distribution from some standardized normal distribution (mean 0 and variance 1)
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]	#randomly creating weights for each layer 


	def feedforward(self,a):
		for b,w in zip(self.biases,self.weights):  #feeding the input to the first layer and then feeding the output of layer 1 to layer 2
			a = sigmoid(np.dot(w,a) + b)  #taking he activation function to be the sigmoid function
		return a #returning the value found at output of last layer

	def stochastic(self,training_data,num_of_epoches,size_of_minibatch,eta,test_data=None) : #function to find the stocastic gradient descent
		if(test_data):
			n_test = len(test_data)

		n = len(training_data)

		for i in xrange(num_of_epoches):
			random.shuffle(training_data)
			minibatches = [training_data[k:k+size_of_minibatch] for k in xrange(0,n,size_of_minibatch)]

			for mini in minibatches:
				self.update_mini_batch(mini,eta)

			#if (test_data):
			#	print "After Epoch ",i,":"
			#	self.testing(test_data)


	def testing(self,test_data):
		results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data] #pair of actual and output we got
		true_results = sum(int(x == y) for (x,y) in results) #number of results which are actually true
		print "Number of correct results : ",true_results


	def update_mini_batch(self,minibatch,eta):
		d_b = [np.zeros(b.shape) for b in self.biases]
		d_w = [np.zeros(w.shape) for w in self.weights]

		for x,y in minibatch:
			grad_b,grad_w = self.backPropagation(x,y)

			d_b = [newb + del_nb for newb,del_nb in zip(d_b,grad_b)]
			d_w = [neww + del_nb for neww,del_nw in zip(d_w,grad_w)]

		self.weights = [w-(eta/len(minibatch))*neww for w,neww in zip(self.weights,d_w)]
		self.biases = [b-(eta/len(minibatch))*newb for b,newb in zip(self.biases,d_b)]



#Backpropagation is about understanding how changing the weights and biases in a network changes the cost function.

	
	def backPropagation(self,x,y):

		grad_b = [np.zeros(b.shape) for b in self.biases] 
		grad_w = [np.zeros(w.shape) for w in self.weights]
		# grad_b and grad_w are both gradient descents for the cost function
		activation = x  # initial activation is basically the input given
		activation_all = [x]   # it will store activation values for each layer
		z_values = []    
		
		for b,w in zip(self.biases,self.weights):
			#print w.shape
			#print activation.shape
			#print b.shape
			z = np.dot(w,activation) + b
			z_values.append(z)
			activation = sigmoid(z)
			activation_all.append(activation)
		
		delta = self.cost_derivative(activation_all[-1],y) *\ sigmoid_slope(z_values[-1])
		grad_b[-1] = delta # according to formula (BP3)
		grad_w[-1] = np.dot(delta,activation_all[-2].transpose()) # accoding to formula (BP4)

		# loop for propagating the delta values backward
		for i in xrange(2,self.num_layers):
			z = z_values[-i]
			sig_der = sigmoid_slope(z)
			#print self.weights[-i+1].transpose().shape
			#print delta.shape
			delta = np.dot(self.weights[-i+1].transpose(),delta)*sig_der       # broadcasting error
			grad_w[-i] = np.dot(delta,activation_all[-i-1].transpose())
			grad_b[-i] = delta

		return (grad_b,grad_w)

	def cost_derivative(self,output_activ,actual_output):  # this method is to find delta(ulta wala) i.e vector of partial derivatives
		# derivative of cost wrt activation is the delta required 
		# we are taking cost function to be quadrartic ((c-a)^2)/2
		# derivative of this is (a-c) which we are returning here
		return (output_activ-actual_output)  

	

training_data,validation_data,test_data = mnist_loader.load_data_wrapper()
net = Network([784,30,10])
#print len(training_data)
net.stochastic(3,30,10,training_data,test_data)
net.stochastic(training_data,30,10,3.0,test_data)










