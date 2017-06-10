

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


h5f = h5py.File('test.h5', 'r')


X = np.asarray(h5f['X'])
Y1 = np.asarray(h5f['Y']).astype(int)


print(Y1)



Y=np.zeros((Y1.shape[0],3))

for i in range(Y1.shape[0]-1):
    Y[i,Y1[i+1]]=1


print(Y)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()

for i in range(X.shape[0]):
    ax.clear()
    ax.imshow(X[i,:,:,:])
    fig.canvas.draw()


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
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    return network


network = input_data(shape=[None, X.shape[1], X.shape[2], X.shape[3]])




network = Alex1(network)


model = tflearn.DNN(network, checkpoint_path='model_alexnet',max_checkpoints=1, tensorboard_verbose=2)




model.fit(X, Y, n_epoch=1, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_step=200,
          snapshot_epoch=False, run_id='Rover_test')


model.save('/home/mpcr/Desktop/TF_Rover/Rover_test.tflearn')




model.load('/home/mpcr/Desktop/TF_Rover/Rover_test.tflearn')




h5f = h5py.File('test.h5', 'r')




X = np.asarray(h5f['X'])


X.shape



for i in range(X.shape[0]):
    s=X[i,:,:,:]
    s=s[None,:,:,:]
    a=model.predict(s)
    a=1*(z==np.max(z,axis=1))
    print(a)






