import numpy as np
import h5py
import tflearn
import matplotlib.pyplot as plt
import scipy.misc
import math
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import tensorflow as tf
tf.reset_default_graph()
#############################################################################################
def Alex1(network):

    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 3, activation='softmax')

    return network
#############################################################################################
h5f = h5py.File('/home/mpcr/Desktop/TF_Rover/test.h5', 'r')
X = np.asarray(h5f['X'])
print(X.shape)
#############################################################################################
network = input_data(shape=[None, X.shape[1], X.shape[2], X.shape[3]])
network = Alex1(network)
model = tflearn.DNN(network)
model.load('/home/mpcr/Desktop/TF_Rover/Rover_test.tflearn')
#############################################################################################


for i in range(X.shape[0]):
    s=X[i,:,:,:]
    s=s[None,:,:,:]
    a=model.predict(s)
    a=1*(a==np.max(a,axis=1))
    print(a)





