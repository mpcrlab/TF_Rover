import os, sys
import tflearn
import tensorflow as tf
import h5py
import numpy as np
from sklearn.preprocessing import scale
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, highway_conv_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization, batch_normalization
from tflearn.layers.estimator import regression
from tflearn import residual_bottleneck, activation, global_avg_pool, merge


def x3(x):
    return 0.3*x**3


def whiten(X):
    '''Function to ZCA whiten image matrix.'''
    
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,V = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    epsilon = 1e-5
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return np.dot(ZCAMatrix, X)


########################################################
def DNN1(network, drop_prob=1.0):
    
    network = tflearn.fully_connected(network, 64, activation='tanh',regularizer='L2', weight_decay=0.001)
    network = tflearn.dropout(network, drop_prob)
    network = tflearn.fully_connected(network, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
    network = tflearn.dropout(network, drop_prob)
    network = tflearn.fully_connected(network, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
    network = tflearn.dropout(network, drop_prob)
    
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network


########################################################
def Conv1(network, drop_prob=1.0):
    
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, drop_prob)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, drop_prob)
    
    network = fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def Alex1(network, drop_prob=1.0):
        
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
    network = dropout(network, drop_prob)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, drop_prob)
    
    network = fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def VGG1(network, drop_prob=1.0):
        
    network = conv_2d(network, 45, 3, activation='relu')
    network = conv_2d(network, 45, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 120, 3, activation='relu')
    network = conv_2d(network, 120, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 200, 3, activation='relu')
    network = conv_2d(network, 200, 3, activation='relu')
    network = conv_2d(network, 200, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 450, 3, activation='relu')
    network = conv_2d(network, 450, 3, activation='relu')
    network = conv_2d(network, 450, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = conv_2d(network, 450, 3, activation='relu')
    network = conv_2d(network, 450, 3, activation='relu')
    network = conv_2d(network, 450, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 3500, activation='relu')
    network = dropout(network, drop_prob)
    network = fully_connected(network, 3500, activation='relu')
    network = dropout(network, drop_prob)
    
    network = fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def Highway1(network, drop_prob=1.0):
        
    dense1 = tflearn.fully_connected(network, 64, activation='elu', regularizer='L2', weight_decay=0.001)

    highway = dense1                              
    for i in range(10):
        highway = tflearn.highway(highway, 64, activation='elu',regularizer='L2', weight_decay=0.001, transform_dropout=0.7)

    network = tflearn.fully_connected(highway, 4, activation='softmax')
    
    return network



########################################################
def ConvHighway1(network, drop_prob=1.0):
        
    for i in range(3):
        for j in [3, 2, 1]: 
            network = highway_conv_2d(network, 16, j, activation='elu')
        network = max_pool_2d(network, 2)
        network = batch_normalization(network)

    network = fully_connected(network, 128, activation='elu')
    network = fully_connected(network, 256, activation='elu')
    
    network = fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def Net_in_Net1(network, drop_prob=1.0):
        
    network = conv_2d(network, 192, 5, activation='relu')
    network = conv_2d(network, 160, 1, activation='relu')
    network = conv_2d(network, 96, 1, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = dropout(network, drop_prob)
    network = conv_2d(network, 192, 5, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = avg_pool_2d(network, 3, strides=2)
    network = dropout(network, drop_prob)
    network = conv_2d(network, 192, 3, activation='relu')
    network = conv_2d(network, 192, 1, activation='relu')
    network = conv_2d(network, 10, 1, activation='relu')
    
    network = avg_pool_2d(network, 8)
    network = flatten(network)
    
    network = fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def ResNet26(network, drop_prob=1.0):
    n = 2 # number of residual blocks per layer
    
    network = tflearn.conv_2d(network, 32, 7, regularizer='L2', strides=2, activation='relu')  
    network = max_pool_2d(network, 3, strides=2)

    network = tflearn.residual_block(network, n, 32, activation='relu')
    network = tflearn.residual_block(network, n, 32, activation='relu')
    network = tflearn.residual_block(network, n, 64, downsample=True, activation='relu')
    network = tflearn.residual_block(network, n, 64, activation='relu')
    network = tflearn.residual_block(network, n, 128, activation='relu')
    network = tflearn.residual_block(network, n, 128, activation='relu')
    network = batch_normalization(network)
    network = activation(network, 'relu')

    network = global_avg_pool(network)
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network


########################################################
def ResNeXt(network):
    c = 32 # cardinality of each residual block
    
    network = tflearn.conv_2d(network, 64, 7, regularizer='L2', strides=2, activation='linear')  
    network = max_pool_2d(network, 3, strides=2)
    network = batch_normalization(network)
    network = activation(network, 'relu')

    network = resnext_block0(network, 2, 128, c, downsample=True)
    network = resnext_block0(network, 2, 256, c, downsample=True)
    network = resnext_block0(network, 2, 512, c, downsample=True)
    network = resnext_block0(network, 2, 1024, c)
    print(network)

    network = global_avg_pool(network)
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def LSTM1(network):
    
    print(network.shape)
    network = tflearn.lstm(network, 500, return_seq=True)
    network = tflearn.lstm(network, 500)
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network



########################################################
def GoogLeNet1(network):
        
    conv1_7_7 = conv_2d(network, 64, 7, strides=2, activation='relu', name = 'conv1_7_7_s2')
    pool1_3_3 = max_pool_2d(conv1_7_7, 3,strides=2)
    pool1_3_3 = local_response_normalization(pool1_3_3)
    conv2_3_3_reduce = conv_2d(pool1_3_3, 64,1, activation='relu',name = 'conv2_3_3_reduce')
    conv2_3_3 = conv_2d(conv2_3_3_reduce, 192,3, activation='relu', name='conv2_3_3')
    conv2_3_3 = local_response_normalization(conv2_3_3)
    pool2_3_3 = max_pool_2d(conv2_3_3, kernel_size=3, strides=2, name='pool2_3_3_s2')
    inception_3a_1_1 = conv_2d(pool2_3_3, 64, 1, activation='relu', name='inception_3a_1_1')
    inception_3a_3_3_reduce = conv_2d(pool2_3_3, 96,1, activation='relu', name='inception_3a_3_3_reduce')
    inception_3a_3_3 = conv_2d(inception_3a_3_3_reduce, 128,filter_size=3,  activation='relu', name = 'inception_3a_3_3')
    inception_3a_5_5_reduce = conv_2d(pool2_3_3,16, filter_size=1,activation='relu', name ='inception_3a_5_5_reduce' )
    inception_3a_5_5 = conv_2d(inception_3a_5_5_reduce, 32, filter_size=5, activation='relu', name= 'inception_3a_5_5')
    inception_3a_pool = max_pool_2d(pool2_3_3, kernel_size=3, strides=1, )
    inception_3a_pool_1_1 = conv_2d(inception_3a_pool, 32, filter_size=1, activation='relu', name='inception_3a_pool_1_1')

    # merge the inception_3a__
    inception_3a_output = merge([inception_3a_1_1, inception_3a_3_3, inception_3a_5_5, inception_3a_pool_1_1], mode='concat', axis=3)

    inception_3b_1_1 = conv_2d(inception_3a_output, 128,filter_size=1,activation='relu', name= 'inception_3b_1_1' )
    inception_3b_3_3_reduce = conv_2d(inception_3a_output, 128, filter_size=1, activation='relu', name='inception_3b_3_3_reduce')
    inception_3b_3_3 = conv_2d(inception_3b_3_3_reduce, 192, filter_size=3,  activation='relu',name='inception_3b_3_3')
    inception_3b_5_5_reduce = conv_2d(inception_3a_output, 32, filter_size=1, activation='relu', name = 'inception_3b_5_5_reduce')
    inception_3b_5_5 = conv_2d(inception_3b_5_5_reduce, 96, filter_size=5,  name = 'inception_3b_5_5')
    inception_3b_pool = max_pool_2d(inception_3a_output, kernel_size=3, strides=1,  name='inception_3b_pool')
    inception_3b_pool_1_1 = conv_2d(inception_3b_pool, 64, filter_size=1,activation='relu', name='inception_3b_pool_1_1')

    #merge the inception_3b_*
    inception_3b_output = merge([inception_3b_1_1, inception_3b_3_3, inception_3b_5_5, inception_3b_pool_1_1], mode='concat',axis=3,name='inception_3b_output')

    pool3_3_3 = max_pool_2d(inception_3b_output, kernel_size=3, strides=2, name='pool3_3_3')
    inception_4a_1_1 = conv_2d(pool3_3_3, 192, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4a_3_3_reduce = conv_2d(pool3_3_3, 96, filter_size=1, activation='relu', name='inception_4a_3_3_reduce')
    inception_4a_3_3 = conv_2d(inception_4a_3_3_reduce, 208, filter_size=3,  activation='relu', name='inception_4a_3_3')
    inception_4a_5_5_reduce = conv_2d(pool3_3_3, 16, filter_size=1, activation='relu', name='inception_4a_5_5_reduce')
    inception_4a_5_5 = conv_2d(inception_4a_5_5_reduce, 48, filter_size=5,  activation='relu', name='inception_4a_5_5')
    inception_4a_pool = max_pool_2d(pool3_3_3, kernel_size=3, strides=1,  name='inception_4a_pool')
    inception_4a_pool_1_1 = conv_2d(inception_4a_pool, 64, filter_size=1, activation='relu', name='inception_4a_pool_1_1')

    inception_4a_output = merge([inception_4a_1_1, inception_4a_3_3, inception_4a_5_5, inception_4a_pool_1_1], mode='concat', axis=3, name='inception_4a_output')


    inception_4b_1_1 = conv_2d(inception_4a_output, 160, filter_size=1, activation='relu', name='inception_4a_1_1')
    inception_4b_3_3_reduce = conv_2d(inception_4a_output, 112, filter_size=1, activation='relu', name='inception_4b_3_3_reduce')
    inception_4b_3_3 = conv_2d(inception_4b_3_3_reduce, 224, filter_size=3, activation='relu', name='inception_4b_3_3')
    inception_4b_5_5_reduce = conv_2d(inception_4a_output, 24, filter_size=1, activation='relu', name='inception_4b_5_5_reduce')
    inception_4b_5_5 = conv_2d(inception_4b_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4b_5_5')

    inception_4b_pool = max_pool_2d(inception_4a_output, kernel_size=3, strides=1,  name='inception_4b_pool')
    inception_4b_pool_1_1 = conv_2d(inception_4b_pool, 64, filter_size=1, activation='relu', name='inception_4b_pool_1_1')

    inception_4b_output = merge([inception_4b_1_1, inception_4b_3_3, inception_4b_5_5, inception_4b_pool_1_1], mode='concat', axis=3, name='inception_4b_output')


    inception_4c_1_1 = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu',name='inception_4c_1_1')
    inception_4c_3_3_reduce = conv_2d(inception_4b_output, 128, filter_size=1, activation='relu', name='inception_4c_3_3_reduce')
    inception_4c_3_3 = conv_2d(inception_4c_3_3_reduce, 256,  filter_size=3, activation='relu', name='inception_4c_3_3')
    inception_4c_5_5_reduce = conv_2d(inception_4b_output, 24, filter_size=1, activation='relu', name='inception_4c_5_5_reduce')
    inception_4c_5_5 = conv_2d(inception_4c_5_5_reduce, 64,  filter_size=5, activation='relu', name='inception_4c_5_5')

    inception_4c_pool = max_pool_2d(inception_4b_output, kernel_size=3, strides=1)
    inception_4c_pool_1_1 = conv_2d(inception_4c_pool, 64, filter_size=1, activation='relu', name='inception_4c_pool_1_1')

    inception_4c_output = merge([inception_4c_1_1, inception_4c_3_3, inception_4c_5_5, inception_4c_pool_1_1], mode='concat', axis=3,name='inception_4c_output')

    inception_4d_1_1 = conv_2d(inception_4c_output, 112, filter_size=1, activation='relu', name='inception_4d_1_1')
    inception_4d_3_3_reduce = conv_2d(inception_4c_output, 144, filter_size=1, activation='relu', name='inception_4d_3_3_reduce')
    inception_4d_3_3 = conv_2d(inception_4d_3_3_reduce, 288, filter_size=3, activation='relu', name='inception_4d_3_3')
    inception_4d_5_5_reduce = conv_2d(inception_4c_output, 32, filter_size=1, activation='relu', name='inception_4d_5_5_reduce')
    inception_4d_5_5 = conv_2d(inception_4d_5_5_reduce, 64, filter_size=5,  activation='relu', name='inception_4d_5_5')
    inception_4d_pool = max_pool_2d(inception_4c_output, kernel_size=3, strides=1,  name='inception_4d_pool')
    inception_4d_pool_1_1 = conv_2d(inception_4d_pool, 64, filter_size=1, activation='relu', name='inception_4d_pool_1_1')

    inception_4d_output = merge([inception_4d_1_1, inception_4d_3_3, inception_4d_5_5, inception_4d_pool_1_1], mode='concat', axis=3, name='inception_4d_output')

    inception_4e_1_1 = conv_2d(inception_4d_output, 256, filter_size=1, activation='relu', name='inception_4e_1_1')
    inception_4e_3_3_reduce = conv_2d(inception_4d_output, 160, filter_size=1, activation='relu', name='inception_4e_3_3_reduce')
    inception_4e_3_3 = conv_2d(inception_4e_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_4e_3_3')
    inception_4e_5_5_reduce = conv_2d(inception_4d_output, 32, filter_size=1, activation='relu', name='inception_4e_5_5_reduce')
    inception_4e_5_5 = conv_2d(inception_4e_5_5_reduce, 128,  filter_size=5, activation='relu', name='inception_4e_5_5')
    inception_4e_pool = max_pool_2d(inception_4d_output, kernel_size=3, strides=1,  name='inception_4e_pool')
    inception_4e_pool_1_1 = conv_2d(inception_4e_pool, 128, filter_size=1, activation='relu', name='inception_4e_pool_1_1')


    inception_4e_output = merge([inception_4e_1_1, inception_4e_3_3, inception_4e_5_5,inception_4e_pool_1_1],axis=3, mode='concat')

    pool4_3_3 = max_pool_2d(inception_4e_output, kernel_size=3, strides=2, name='pool_3_3')


    inception_5a_1_1 = conv_2d(pool4_3_3, 256, filter_size=1, activation='relu', name='inception_5a_1_1')
    inception_5a_3_3_reduce = conv_2d(pool4_3_3, 160, filter_size=1, activation='relu', name='inception_5a_3_3_reduce')
    inception_5a_3_3 = conv_2d(inception_5a_3_3_reduce, 320, filter_size=3, activation='relu', name='inception_5a_3_3')
    inception_5a_5_5_reduce = conv_2d(pool4_3_3, 32, filter_size=1, activation='relu', name='inception_5a_5_5_reduce')
    inception_5a_5_5 = conv_2d(inception_5a_5_5_reduce, 128, filter_size=5,  activation='relu', name='inception_5a_5_5')
    inception_5a_pool = max_pool_2d(pool4_3_3, kernel_size=3, strides=1,  name='inception_5a_pool')
    inception_5a_pool_1_1 = conv_2d(inception_5a_pool, 128, filter_size=1,activation='relu', name='inception_5a_pool_1_1')

    inception_5a_output = merge([inception_5a_1_1, inception_5a_3_3, inception_5a_5_5, inception_5a_pool_1_1], axis=3,mode='concat')


    inception_5b_1_1 = conv_2d(inception_5a_output, 384, filter_size=1,activation='relu', name='inception_5b_1_1')
    inception_5b_3_3_reduce = conv_2d(inception_5a_output, 192, filter_size=1, activation='relu', name='inception_5b_3_3_reduce')
    inception_5b_3_3 = conv_2d(inception_5b_3_3_reduce, 384,  filter_size=3,activation='relu', name='inception_5b_3_3')
    inception_5b_5_5_reduce = conv_2d(inception_5a_output, 48, filter_size=1, activation='relu', name='inception_5b_5_5_reduce')
    inception_5b_5_5 = conv_2d(inception_5b_5_5_reduce,128, filter_size=5,  activation='relu', name='inception_5b_5_5' )
    inception_5b_pool = max_pool_2d(inception_5a_output, kernel_size=3, strides=1,  name='inception_5b_pool')
    inception_5b_pool_1_1 = conv_2d(inception_5b_pool, 128, filter_size=1, activation='relu', name='inception_5b_pool_1_1')
    inception_5b_output = merge([inception_5b_1_1, inception_5b_3_3, inception_5b_5_5, inception_5b_pool_1_1], axis=3, mode='concat')

    pool5_7_7 = avg_pool_2d(inception_5b_output, kernel_size=7, strides=1)
    pool5_7_7 = dropout(pool5_7_7, 0.4)
    
    network = fully_connected(pool5_7_7, 4,activation='softmax')

    return network

########################################################
def DenseNet(network):
        
    # Growth Rate (12, 16, 32, ...)
    k = 3

    # Depth (40, 100, ...)
    L = 28
    nb_layers = int((L - 4) / 3)

    # Building DenseNet Network
    
    network = tflearn.conv_2d(network, 10, 4, regularizer='L2', weight_decay=0.0001)
    network = denseblock(network, nb_layers, k, dropout=drop_prob)
    network = denseblock(network, nb_layers, k, dropout=drop_prob)
    network = denseblock(network, nb_layers, k, dropout=drop_prob)
    network = tflearn.global_avg_pool(network)

    # Regression
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network
    
########################################################
def RCNN1(network, prev_activation=None, scale=False):
    if prev_activation is None:
        prev_activation = tf.zeros([1, 2500])
    
    if scale is True:
        network = tf.transpose(tf.reshape(network, [-1, num_rows*num_cols*num_channels]))
        mean, var = tf.nn.moments(network, [0])
        network = tf.transpose((network-mean)/(tf.sqrt(var)+1e-6))
        network = tf.reshape(network, [-1, num_rows, num_cols, num_channels])
        
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
    network = fully_connected(network, 2500, activation='tanh')
    network = dropout(network, drop_prob)
    feat_layer = fully_connected(network, 2500, activation='tanh')
    network = merge([feat_layer, prev_activation], 'concat', axis=1)
    network = lstm(network, 250, dropout=drop_prob, activation='relu')
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network, feat_layer

########################################################
def lstm2(network):
    network = lstm(network, 10000, dropout=0.7, activation='relu')
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network 

########################################################
def X3(y, iters, batch_sz, num_dict_features=None, D=None):
    ''' Dynamical systems neural network used for sparse approximation of an
        input vector.
        
        Args: 
            y: input signal or vector, or multiple column vectors.
            num_dict_features: number of dictionary patches to learn.
            iters: number of LCA iterations.
            batch_sz: number of samples to send to the network at each iteration.
            D: The dictionary to be used in the network.'''
  
    assert(num_dict_features is None or D is None), 'provide D or num_dict_features, not both'
    
    e = np.zeros([iters, 1])
    D=np.random.randn(y.shape[0], num_dict_features)

    for i in range(iters):
        # choose random examples this iteration
        batch=y[:, np.random.randint(0, y.shape[1], batch_sz)]
        batch = scale(batch, 1)
        batch = whiten(batch)

        # scale the values in the dict to between 0 and 1
        D=np.matmul(D, np.diag(1/(np.sqrt(np.sum(D**2, 0))+1e-6)))
        # get similarity between each feature and each data patch
        a = np.matmul(D.transpose(), batch)
        # scale the alpha coefficients (cosine similarity coefficients)
        a=np.matmul(a, np.diag(1/(np.sqrt(np.sum(a**2, 0))+1e-6)))
        # perform cubic activation on the alphas
        a=0.5*a**3
        # get the SSE between reconstruction and data batch
        error = batch - np.matmul(D, a)
        # save the error to plot later
        e[i, 0] = np.mean(error**2)
        # modify the dictionary to reduce the error
        D=D+np.matmul(error, a.transpose())

    return D, a, e

######################################################################
def ResNet34(network):
    
    network = tflearn.conv_2d(network, 64, 7, 
                              strides=2, activation='linear',
                              regularizer='L2')  
    network = max_pool_2d(network, 3, strides=2)
    
    network = tflearn.residual_block(network, 3, 64, activation='relu')
    network = tflearn.residual_block(network, 1, 128, activation='relu', downsample=True)
    network = tflearn.residual_block(network, 3, 128, activation='relu')
    network = tflearn.residual_block(network, 1, 256, activation='relu', downsample=True)
    network = tflearn.residual_block(network, 5, 256, activation='relu')
    network = tflearn.residual_block(network, 1, 512, activation='relu', downsample=True)
    network = tflearn.residual_block(network, 2, 512, activation='relu')
    network = batch_normalization(network)
    network = activation(network, 'relu')
    
    print(network)
    network = global_avg_pool(network)
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network



#######################################################################
def ResNeXt34(network):
    c = 8  #cardinality
    
    network = tflearn.conv_2d(network, 32, 7, strides=2, activation='linear')  
    network = max_pool_2d(network, 3, strides=2)
    network = batch_normalization(network)
    network = activation(network, 'relu')

    network = resnext_block5(network, 3, 32, c)
    network = resnext_block5(network, 1, 64, c, downsample=True)
    network = resnext_block5(network, 3, 64, c)
    network = resnext_block5(network, 1, 128, c, downsample=True)
    network = resnext_block5(network, 5, 128, c)
    network = resnext_block5(network, 1, 256, c, downsample=True)
    network = resnext_block5(network, 2, 256, c)

    
    network = global_avg_pool(network)
    network = tflearn.fully_connected(network, 4, activation='softmax')
    
    return network
    
##########################################################################
##########################################################################
##########################################################################
    
    
modelswitch = {
    0 : DNN1,
    1 : Conv1,
    2 : Alex1,
    3 : VGG1,
    4 : Highway1,
    5 : ConvHighway1,
    6 : Net_in_Net1,
    7 : ResNet26,
    8 : ResNeXt,
    9 : GoogLeNet1,
    10 : LSTM1,
    11 : DenseNet,
    12 : RCNN1,
    13 : lstm2,
    14 : X3,
    15: ResNet34,
    16: ResNeXt34,
}


