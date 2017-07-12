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
import cv2
from NetworkSwitch import *

print('What filename do you want to save this model as?')
m_save = raw_input('Rover Used_number of frames/stackinterval_other parameters  ')

model_num = np.int32(raw_input('Which model do you want to train (0 - 10)?'))

os.chdir('/home/TF_Rover/RoverData')
fnames = glob.glob('*.h5')
epochs = 800
batch_sz = 70
errors = []
test_num = 600
f_int = 10
f_int2 = 20
val_accuracy = []
num_stack = 3
val_name = 'Run_2_both_lights_on.h5'



def add_noise(x, y):
    x_aug = x + 0.7 * np.random.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    x = np.concatenate((x, x_aug), 0)
    y = np.concatenate((y, y), 0)
    return x, y


# Validation set
print('Validation Dataset: %s'%(val_name))

# load the h5 file containing the data used for validation
val_file = h5py.File(val_name, 'r')
tx = np.asarray(val_file['X'])
y_ = np.int32(np.asarray(val_file['Y']) + 1.)

# crop and grayscale the validation images, and select 1000
tx = np.mean(tx[:test_num, 110:, :, :], 3, keepdims=True)
ty = np.zeros([test_num, 3])
ty[np.arange(test_num), y_[:test_num]] = 1.

# Feature Scaling validation data
tx = np.transpose(tx.reshape([test_num, -1]))
tx = (tx-np.mean(tx, 0))/(np.std(tx, 0)+1e-6)
tx = np.reshape(tx.transpose(), [test_num, 130, 320, 1])

# Create validation framestack
TX = []
TY = []
for i in range(test_num-1, f_int2, -1):
    TX2 = tx[i-f_int, :, :, :]
    TX3 = tx[i-f_int2, :, :, :]
    TX.append(np.concatenate((tx[i, :, :, :], TX2, TX3), 2))
    TY.append(ty[i, :])

TX = np.asarray(TX)
TY = np.asarray(TY)
print(TX.shape)
print(TY.shape)
assert(TY.shape[0] == TX.shape[0]),'data and label shapes do not match'

labels = tf.placeholder(dtype=tf.float32, shape=[None, 3])
network = tf.placeholder(dtype=tf.float32, shape=[None, 130, 320, num_stack])


net_out = modelswitch[model_num](network)
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(net_out, 1), tf.argmax(labels, 1))))
cost = categorical_crossentropy(net_out, labels)
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
trainop = tflearn.TrainOp(loss=cost,
                         optimizer=opt,
                         metric=None,
                         batch_size=batch_sz)
model = Trainer(train_ops=trainop)


for i in range(epochs):
    # pick random dataset for this epoch
    n = np.random.randint(1, len(fnames)-1, 1)
    filename = fnames[n[0]]
    
    if filename == val_name:
        print('skipping iteration')
        continue
        
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    y = np.int32(f['Y']) + 1
    Y = np.zeros([y.shape[0], 3])
    #rand = np.random.randint(0, X.shape[0], X.shape[0])
    Y[np.arange(Y.shape[0]), y] = 1.0
    X = np.mean(X[:, 110:, :, :], 3, keepdims=True)
    assert(X.shape[0] == Y.shape[0]), 'Data/label dimensions not equal'
    num_batches = np.int32(np.ceil(X.shape[0]/batch_sz))

    for j in range(num_batches):
        if j * batch_sz + batch_sz <= (X.shape[0]-1):
            x = X[j*batch_sz:j*batch_sz+batch_sz, :, :, :]
            y = Y[j*batch_sz:j*batch_sz+batch_sz, :]
        elif j*batch_sz + batch_sz >= (X.shape[0]-1) and X.shape[0] - (j*batch_sz + batch_sz) >= f_int2:
            x = X[j*batch_sz:X.shape[0], :, :, :]
            y = Y[j*batch_sz:X.shape[0], :]
        else:
            continue

        # Feature Scaling
        x = np.transpose(x.reshape([y.shape[0], -1]))
        x = (x-np.mean(x, 0))/(np.std(x, 0)+1e-6)
        x = np.reshape(x.transpose(), [y.shape[0], 130, 320, 1])


        # Create framestack
        X_ = []
        Y_ = []

        for ex_num in range(x.shape[0]-1, f_int2, -1):
            X2 = x[ex_num-f_int, :, :, :]
            X3 = x[ex_num-f_int2, :, :, :]
            X_.append(np.concatenate((x[ex_num, :, :, :], X2, X3), 2))
            Y_.append(y[ex_num, :])


        # Data Augmentation
        X_, Y_ = add_noise(np.asarray(X_), np.asarray(Y_))
            
        # Training
        model.fit_batch(feed_dicts={network:X_, labels:Y_})
        train_acc = model.session.run(acc, feed_dict={network:X_, labels:Y_})
        sys.stdout.write('Epoch %d; dataset %s; train_acc: %.2f; loss: %f  \r'%(
                                    i+1, filename, train_acc, 1-train_acc) )
        sys.stdout.flush()

    val_acc = model.session.run(acc, feed_dict={network:TX, labels:TY})
    print(val_acc)
    errors.append(1.-val_acc)
    val_accuracy.append(val_acc)
    f.close()

np.save(m_save+modelswitch[model_num].__name__+'.npy', errors, val_accuracy)
model.save(m_save+modelswitch[model_num].__name__)

fig = plt.figure()
a1 = fig.add_subplot(111)
a1.plot(errors)
plt.show()
