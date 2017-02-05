import cPickle
import gzip
import numpy as np

# training data is a list of tuples of input and output(actual).
# The first entry of each tuple is 28*28 ndarray. There are a total of 50000
# elements int this list.Similar is for test and validation data except the
# number of elements is 10000

def loading():
	file = gzip.open('mnist.pkl.gz')
	trainig,validation,test = cPickle.load(file)
	file.close()
	return (trainig,validation,test)


# return datasets obtained from loading() function after converting it into
# a format more suitable for out neural network
def load_data_wrapper():
	training,validation,test = loading()

	train_inputs = [np.reshape(x,(784,1)) for x in training[0]]
	train_results = [vectorisation(y) for y in training[1]]

	valid_inputs = [np.reshape(x,(784,1)) for x in validation[0]]
	#valid_results = [vectorisation(y) for y in validation[1]]

	test_inputs = [np.reshape(x,(784,1)) for x in test[0]]
	#test_results = [vectorisation(y) for y in test[1]]

	training_data = zip(train_inputs,train_results)
	validation_data = zip(valid_inputs,validation[1])
	test_data = zip(test_inputs,test[1])

	return (training_data,validation_data,test_data)


def vectorisation(j):
	#return 10-dimensional array containing all zeroes except at index 
	#which is equal to that particular digit
	vec = np.zeros((10,1))
	vec[j] = 1.0
	return vec


#training_data,validation_data,test_data = load_data_wrapper()
#training_data,validation_data,test_data = loading()