import operator
import threading
import numpy as np
import copy

class ParameterServer(object):

    def __init__(self, sizes):
        self.lock = threading.Lock()        
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]


    def upload_gradient(self, local_weights, local_biases):
        
        self.lock.acquire()
        
        self.weights = [(w + nw)/2
                        for w, nw in zip(self.weights, local_weights)]
        

        self.biases = [(b + nb)/2
                       for b, nb in zip(self.biases, local_biases)]
        
        self.lock.release()
        
        
    def download_gradient(self):

        self.lock.acquire()
        
        global_weights = copy.deepcopy(self.weights)
        global_biases = copy.deepcopy(self.biases)                 

        self.lock.release()
        return global_weights, global_biases


