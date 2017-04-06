import operator
import threading
import numpy

class ParameterServer(object):

    def __init__(self):
        self.num_layers = 2
        self.num_neuron_layer0 = 30
        self.num_neuron_layer1 = 10
        self.num_input = 784
        self.global_gradient_dict = dict()
        self.stats = dict()
        self.lock = threading.Lock()
        
        for layer in range(self.num_layers):
            if layer == 0:
                for neuron in range(self.num_neuron_layer0):
                    for input_parameter in range(self.num_input):
                        key = str(layer) + "_" + str(neuron) + "_" +str(input_parameter) 
                        value = numpy.random.rand(1)[0]
                        self.global_gradient_dict[key] = value
                        self.stats[key] = 0 
            elif layer == 1:
                for neuron in range(self.num_neuron_layer1):
                    for input_parameter in range(self.num_neuron_layer0):
                        key = str(layer) + "_" + str(neuron) + "_" +str(input_parameter)
                        value = numpy.random.rand(1)[0]                  
                        self.global_gradient_dict[key] = value
                        self.stats[key] = 0 
                        print key
                        
                        
            #self.gradient_list.append(gradient_tuple)            
                        
        print self.global_gradient_dict


    def upload_gradient(self, gradient_list):
        
        self.lock.acquire()
        for gradient in gradient_list:
            key = gradient[0]
            value = gradient[1]
            
            self.global_gradient_dict[key] += value
            self.stats[key] += 1
        
        #print self.stats
        self.lock.release()
        
        
    def download_gradient(self, num_gradient_download):
        sorted_stats = sorted(self.stats.items(), key=operator.itemgetter(1), reverse=True)
        gradient_download_list = list()      
        self.lock.acquire()                
        
        for counter in range(num_gradient_download):
            stats_tuple = sorted_stats[counter]
            key = stats_tuple[0]
            gradient_element = (key, self.global_gradient_dict[key])
            gradient_download_list.append(gradient_element)
            
        self.lock.release()
        return gradient_download_list


