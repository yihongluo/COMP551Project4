import cPickle
import threading
import parameter_server
import gzip
import numpy as np
from matplotlib import pyplot as plt
import neural_network


def get_data():
    dataFile = gzip.open('mnist.pkl.gz', 'rb')
    t1, v1, t2 = cPickle.load(dataFile)
    dataFile.close()
    
    tr_inputs = [np.reshape(x, (784, 1)) for x in t1[0]]
      
 

    tr_outputs = []
    for y in t1[1]:
        o = np.zeros((10, 1))
        o[y] = 1.0
        tr_outputs.append(o)
        
        
        
    print tr_outputs[10]
    
    tr_data = zip(tr_inputs, tr_outputs)
    vd_inputs = [np.reshape(x, (784, 1)) for x in v1[0]]
    v_data = zip(vd_inputs, v1[1])
    t_inputs = [np.reshape(x, (784, 1)) for x in t2[0]]
    testing_set = zip(t_inputs, t2[1])
    return (tr_data, v_data, testing_set)




def participant_manager():
    train, v1, test = get_data()
    
    server = parameter_server.ParameterServer([784, 30, 10])
    
    start = 0
    end = start+1000
    for i in range(20):
        tr_data_participant = train[start:end]        
        participant = threading.Thread(name='participant' + str(i), target=ffn, args = (tr_data_participant, test, server))
        participant.start()
        start = end
        end = end + 1000
        
        

        
    
    
def ffn(tr_data, testing_set, server):
    #1 is the upload ratio
    net = neural_network.NeuralNetwork([784, 30, 10], 1, 1, server)
    #net = network.Network([784, 30, 10], 0.1, 0.1, server)
    net.train_network(tr_data, 50, 1, 2, testing_set)
    
    
participant_manager()  
    