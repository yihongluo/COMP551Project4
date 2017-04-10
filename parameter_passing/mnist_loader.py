import cPickle
import threading

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
    
    #server = parameter_server.ParameterServer()
    
    start = 0
    end = start+1000
    existing_weights = None
    existing_bias = None
    for i in range(20):
        collaborate = False
        weights = None
        bias = None
        
        if i > 0:
            collaborate = True
            weights = existing_weights
            bias = existing_bias
            
        training_data_participant = train[start:end]        
        existing_weights, existing_bias = ffn(training_data_participant, test, collaborate, weights, bias)
        
        start = end
        end = end + 1000
        #break
# 	
# 		
# def participant_manager1():
#     training_data, validation_data, test_data = load_data_wrapper()
#     
#     
#     training_data_participant1 = training_data[:5000]
#     training_data_participant2 = training_data[5000:10000]
#     training_data_participant3 = training_data[10000:15000]
#     training_data_participant4 = training_data[15000:20000]
#     training_data_participant5 = training_data[20000:25000]
#     training_data_participant6 = training_data[25000:30000]
#     training_data_participant7 = training_data[30000:35000]
#     training_data_participant8 = training_data[35000:40000]
#     training_data_participant9 = training_data[40000:45000]
#     training_data_participant10 = training_data[45000:50000]
#         
#     participant1 = threading.Thread(name='participant1', target=ffn, args = (training_data_participant1, test_data))
#     participant2 = threading.Thread(name='participant2', target=ffn, args = (training_data_participant2, test_data))
#     participant3 = threading.Thread(name='participant3', target=ffn, args = (training_data_participant3, test_data))
#     participant4 = threading.Thread(name='participant4', target=ffn, args = (training_data_participant4, test_data))
#     participant5 = threading.Thread(name='participant5', target=ffn, args = (training_data_participant5, test_data))
#     participant6 = threading.Thread(name='participant6', target=ffn, args = (training_data_participant6, test_data))
#     participant7 = threading.Thread(name='participant7', target=ffn, args = (training_data_participant7, test_data))
#     participant8 = threading.Thread(name='participant8', target=ffn, args = (training_data_participant8, test_data))
#     participant9 = threading.Thread(name='participant9', target=ffn, args = (training_data_participant9, test_data))
#     participant10 = threading.Thread(name='participant10', target=ffn, args = (training_data_participant10, test_data))
#     
#     
#     
#     participant1.start()
#     participant2.start()
#     participant3.start()
#     participant4.start()
#     participant5.start()
#     participant6.start()
#     participant7.start()
#     participant8.start()
#     participant9.start()
#     participant10.start()  	
    
    
    
def ffn(tr_data, testing_set, collaborate, weights, bias):
    #0.1 is the upload ratio
    net = neural_network.NeuralNetwork([784, 30, 10], collaborate, weights, bias)
    final_weights, final_biases = net.train_network(tr_data, 50, 1, 0.1, testing_set)
    return final_weights, final_biases
    
    
participant_manager()
    
    