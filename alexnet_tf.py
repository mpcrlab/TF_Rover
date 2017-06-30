from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.helpers.trainer import Trainer
import h5py
from tflearn.metrics import *
from tflearn.objectives import categorical_crossentropy
import glob
import matplotlib.pyplot as plt
from tflearn.data_augmentation import ImageAugmentation
import sys
import os


os.chdir('/home/TF_Rover/RoverData')
fnames = glob.glob('*.h5')
epochs = 600
batch_sz = 64
errors = []
test_num = 850


def add_noise(x, y):
    x_aug = x + 0.7 * np.random.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    x = np.concatenate((x, x_aug), 0)
    y = np.concatenate((y, y), 0)
    return x, y


# Validation set
print('Validation Dataset: %s'%(fnames[-1]))
val_file = h5py.File(fnames[-1], 'r')
tx = np.asarray(val_file['X'])
y_ = np.int32(np.asarray(val_file['Y']) + 1.)
R = np.random.randint(0, tx.shape[0], test_num)
tx = np.mean(tx[R, 110:, :, :], 3, keepdims=True)
ty = np.zeros([test_num, 3])
ty[np.arange(test_num), y_[R]] = 1.
assert(np.sum(ty) == ty.shape[0]), 'more than one label per example'

# Feature Scaling validation data
tx = np.transpose(tx.reshape([test_num, -1]))
tx = (tx-np.mean(tx, 0))/(np.std(tx, 0)+1e-6)
tx = np.reshape(tx.transpose(), [test_num, 130, 320, 1])


labels = tf.placeholder(dtype=tf.float32, shape=[None, 3])
data = tf.placeholder(dtype=tf.float32, shape=[None, 130, 320, 1])

# Building 'AlexNet'
network = conv_2d(data, 96, 11, strides=4, activation='relu')
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
#network = regression(network, optimizer='momentum',
#                  loss='categorical_crossentropy',
#                  learning_rate=0.001)

acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(network, 1), tf.argmax(labels, 1))))
cost = categorical_crossentropy(network, labels)
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
trainop = tflearn.TrainOp(loss=cost,
                         optimizer=opt,
                         metric=None,
                         batch_size=batch_sz)
model = Trainer(train_ops=trainop)

#model.session.run(tf.initialize_all_variables())
#trainer = tf.train.AdamOptimizer(1e-4).minimize(cost)
#equal = tf.equal(tf.argmax(network, 1), tf.argmax(labels, 1))
#acc = tf.reduce_mean(tf.cast(equal, tf.float32))

for i in range(epochs):
    # pick random dataset for this epoch
    n = np.random.randint(1, len(fnames)-2, 1)
    filename = fnames[n[0]]
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    y = np.int32(f['Y']) + 1
    Y = np.zeros([y.shape[0], 3])
    rand = np.random.randint(0, X.shape[0], X.shape[0])
    Y[np.arange(Y.shape[0]), y[rand]] = 1.0
    X = np.mean(X[rand, 110:, :, :], 3, keepdims=True)
    assert(X.shape[0] == Y.shape[0]), 'Data/label dimensions not equal'
    num_batches = np.int32(np.ceil(X.shape[0]/batch_sz))

    for j in range(num_batches):
        if j * batch_sz + batch_sz <= (X.shape[0]-1):
            x = X[j*batch_sz:j*batch_sz+batch_sz, :, :, :]
            y = Y[j*batch_sz:j*batch_sz+batch_sz, :]
        else:
            x = X[j*batch_sz:X.shape[0]-1, :, :, :]
            y = Y[j*batch_sz:X.shape[0]-1, :]

        # Feature Scaling
        x = np.transpose(x.reshape([y.shape[0], -1]))
        x = (x-np.mean(x, 0))/(np.std(x, 0)+1e-6)
        x = np.reshape(x.transpose(), [y.shape[0], 130, 320, 1])

        # Data Augmentation
        x, y = add_noise(x, y)

        model.fit_batch(feed_dicts={data:x, labels:y})
        train_acc = model.session.run(acc, feed_dict={data:x, labels:y})
        sys.stdout.write('Epoch %d; dataset %s; train_acc: %.2f; loss: %f  \r'%(
                                i+1, filename, train_acc, 1-train_acc) )
        sys.stdout.flush()

    val_acc = model.session.run(acc, feed_dict={data:tx, labels:ty})
    print(val_acc)
    errors.append(1.-val_acc)

model.save('Alexnet_gray_oneframe_both_ways_randomAug')

fig = plt.figure()
a1 = fig.add_subplot(111)
a1.plot(errors)
plt.show()
