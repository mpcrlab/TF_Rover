import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
import matplotlib.pyplot as plt
import scipy

network = input_data(shape=[None, 130, 320, 1])
fc1 = tflearn.fully_connected(network, 64, activation='tanh',regularizer='L2', weight_decay=0.001)
network = tflearn.dropout(fc1, 1.0)
fc2 = tflearn.fully_connected(network, 64, activation='tanh', regularizer='L2', weight_decay=0.001)
network = tflearn.dropout(fc2, 1.0)
network = tflearn.fully_connected(network, 3, activation='softmax')


model = tflearn.DNN(network)
model.load('Felix_1frame_GrayCropped_RightAllDrivers_DNN1', weights_only=True)
a = model.get_weights(fc1.W)

for i in xrange(a.shape[1]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.imshow(a[:, i].reshape([130, 320]))
    plt.show()
