"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import cPickle
import gzip
from matplotlib import pyplot as plt
import network
import threading


# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.
    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.
    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.
    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.
    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    
#     for x in tr_d[0]:
#         print x.shape
        
    print len(training_inputs)
    imageSample = training_inputs[10]
    plt.imshow(np.reshape(imageSample, (28,28)))
    plt.show()
    
    training_results = [vectorized_result(y) for y in tr_d[1]]
    print training_results[10]
    
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def participant_manager():
    training_data, validation_data, test_data = load_data_wrapper()
    
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
            
        training_data_participant = training_data[start:end]        
        existing_weights, existing_bias = ffn(training_data_participant, test_data, collaborate, weights, bias)
        
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
    
    
    
def ffn(training_data, test_data, collaborate, weights, bias):
    #0.1 is the upload ratio
    net = network.Network([784, 30, 10], collaborate, weights, bias)
    final_weights, final_biases = net.SGD(training_data, 30, 1, 0.1, test_data=test_data)
    return final_weights, final_biases
    
    
participant_manager()
    
    