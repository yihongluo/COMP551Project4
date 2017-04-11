
import random
import sys
import numpy as np
import threading
import operator
import copy



import matplotlib.pyplot as plt

class NeuralNetwork(object):

    def __init__(self, dimension, upload_ratio, download_ratio, server):
        self.upload_ratio = upload_ratio
        self.download_ratio = download_ratio
        self.server = server
        self.dimension = dimension
        self.accuracies = []
        self.totalLayers = len(dimension)



    
    #pass training data, testing data, bs , epoch and learning rate
    def train_network(self, train_data, epochs, mini_batch_size, learning_rate, test_data):
        test_set_size = len(test_data)
        train_set_size = len(train_data)
        
        
        for j in xrange(epochs):
            random.shuffle(train_data)
            small_set_list = [train_data[m:m+mini_batch_size] for m in xrange(0, train_set_size, mini_batch_size)]
            #print len(mini_batches)
            
            self.download_weight_parameters()
            old_weights = self.weights
            old_bias = self.biases            
            
            
            for small_data in small_set_list:
                temp_biases = [np.zeros(bias.shape) for bias in self.biases]
                temp_weights = [np.zeros(weight.shape) for weight in self.weights]
        
                for train_input, train_output in small_data:
                    diff_term_temp_biases, diff_term_temp_weights = self.back_propagation(train_input, train_output)
                    temp_biases = [new_bias+diff_new_bias for new_bias, diff_new_bias in zip(temp_biases, diff_term_temp_biases)]
                    temp_weights = [new_weight+diff_new_weight for new_weight, diff_new_weight in zip(temp_weights, diff_term_temp_weights)]
        
        
        
                self.weights = [wt-(learning_rate/len(small_data))*new_wt
                        for wt, new_wt in zip(self.weights, temp_weights)]
        

                self.biases = [bias-(learning_rate/len(small_data))*new_bias
                       for bias, new_bias in zip(self.biases, temp_biases)]
            
            
            
                
            print "Result in {0}: {1} / {2}".format(j, self.get_results(test_data), test_set_size) + " " + threading.currentThread().getName()
            self.accuracies.append(float(self.get_results(test_data)/test_set_size))
                
                
            self.calculate_weights()
            
        #plt.plot(self.accuracies)
        #plt.show()    
            


        


    #downloads weights from the server
    def download_weight_parameters(self):
        global_weights, global_biases = self.server.download_gradient()
        self.weights = global_weights
        self.biases = global_biases    
            
        
        
    #Uploads weights to the server
    def calculate_weights(self):
        local_weights = copy.deepcopy(self.weights)
        local_biases =copy.deepcopy(self.biases)
        
        local_weights = [w + 0.002
                        for w in local_weights]
       
        local_biases = [b + 0.002
                        for b in local_biases]
       
        self.server.upload_gradient(self.weights, self.biases)
            
            
            

    #performs forward propagation and backpropagation 
    def back_propagation(self, data_input, data_output):
        temp_biases = [np.zeros(b.shape) for b in self.biases]
        temp_weights = [np.zeros(w.shape) for w in self.weights]       
        

        temp_input = data_input
        temp_inputs = [data_input]
        temp_output = []
        for bias, weight in zip(self.biases, self.weights):     
            output = np.dot(weight, temp_input)+bias       
            temp_output.append(output)
            temp_input = self.sigmoid_function(output)    
            temp_inputs.append(temp_input)

            
        diff_term = (temp_inputs[-1] - data_output) * (self.sigmoid_function(temp_output[-1]) * (1-self.sigmoid_function(temp_output[-1])))

        temp_biases[-1] = diff_term
        temp_weights[-1] = np.dot(diff_term, temp_inputs[-2].transpose())
        
        for l in xrange(2, self.totalLayers):
            e = temp_output[-l]
            sigmoid_result = self.sigmoid_function(e)*(1-self.sigmoid_function(e))            
            diff_term = np.dot(self.weights[-l+1].transpose(), diff_term) * sigmoid_result
            temp_biases[-l] = diff_term
            temp_weights[-l] = np.dot(diff_term, temp_inputs[-l-1].transpose())
        return (temp_biases, temp_weights)



    
    #Test data is predicted using this function
    def get_results(self, test_data):


        test_results = []
        for input, output in test_data:
            for bias, weight in zip(self.biases, self.weights):
                input = self.sigmoid_function(np.dot(weight, input)+bias)
            output_result = input
            
            test_results.append((np.argmax(output_result), output))
            
            

        return sum(int(actual == expected) for (actual, expected) in test_results)



    def sigmoid_function(self, e):    
        return 1.0/(1.0+np.exp(-e))

