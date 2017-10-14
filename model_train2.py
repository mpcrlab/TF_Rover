# author = michael teti

from __future__ import division, print_function, absolute_import
import numpy as np
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
import sys, os
import cv2
sys.argv += [0.5]
from NetworkSwitch import *
from sklearn.preprocessing import scale
from scipy.misc import imshow


m_save = 'Felix_3frames5,15_GrayCropped_RightAllDrivers_'

# prompt the user for which model they want to train from NetworkSwitch.py
print(modelswitch)
model_num = np.int32(raw_input('Which model do you want to train (0 - 12)?'))

# define useful variables
os.chdir('/home/TF_Rover/RoverData/Right2')
fnames = glob.glob('*.h5') # datasets to train on
epochs = 20001 # number of training iterations
batch_sz = 80  # training batch size
f_int = 5
f_int2 = 15
num_stack = 3
val_name = 'Run_218seconds_Michael_Sheri.h5' # Dataset to use for validation
num_classes = 4


def add_noise(x, y):
    x_aug = x + np.random.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    x = np.concatenate((x, x_aug), 0)
    y = np.concatenate((y, y), 0)
    return x, y


def create_framestack(x, y, f_int, f_int2):
    X_ = []
    Y_ = []
    for ex_num in range(x.shape[0]-1, f_int2, -1):
        X2 = x[ex_num-f_int, :, :, :]
        X3 = x[ex_num-f_int2, :, :, :]
        X_.append(np.concatenate((x[ex_num, :, :, :], X2, X3), 2))
        Y_.append(y[ex_num, :])
    return np.asarray(X_), np.asarray(Y_)


def feature_scale(x):
    x = scale(x.reshape([x.shape[0], -1]), 1)
    return x.reshape([x.shape[0], 130, 320, 1])


def batch_get(filename, batch_size):
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    y = np.int32(f['Y']) + 1
    print(y.shape)
    rand = np.random.randint(f_int2, X.shape[0], batch_size)
    Y = np.zeros([batch_size, num_classes])
    Y[np.arange(batch_size), y[rand]] = 1.0
    X = np.mean(X[rand, 110:, :, :], 3, keepdims=True) # grayscale and crop frames
    assert(X.shape[0] == Y.shape[0]), 'Data and labels different sizes'
    f.flush()
    f.close()
    return X, Y


# Validation set
print('Validation Dataset: %s'%(val_name))

# Create input layer and label placeholder for the network
labels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
network = tf.placeholder(dtype=tf.float32, shape=[None, 130, 320, num_stack])


# send the input placeholder to the specified network
net_out = modelswitch[model_num](network)
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(net_out, 1), tf.argmax(labels, 1))))
cost = categorical_crossentropy(net_out, labels) # crossentropy loss function

# Tensorboard summaries
tf.summary.scalar('Accuracy_', acc)
tf.summary.scalar('Loss_', cost)
merged = tf.summary.merge_all()


# gradient descent optimizer
opt = tf.train.AdamOptimizer(learning_rate=8e-6)
trainop = tflearn.TrainOp(loss=cost,
                         optimizer=opt,
                         metric=None,
                         batch_size=batch_sz)
model = Trainer(train_ops=trainop)

writer = tf.summary.FileWriter('/tmp/tflearn_logs/test'+m_save+modelswitch[model_num].__name__,
                               model.session.graph)
writer2 = tf.summary.FileWriter('/tmp/tflearn_logs/train'+m_save+modelswitch[model_num].__name__,
                               model.session.graph)



################################## Main Loop #######################################

for i in range(epochs):
    
    # pick random dataset for this epoch
    n = np.random.randint(1, len(fnames)-1, 1)
    filename = fnames[n[0]]
    
    # skip validation set if chosen
    if filename == val_name: 
        continue

    # load the chosen data file
    X, Y = batch_get(filename, batch_sz)
    
    sys.exit(0)

    # local feature Scaling
    X = feature_scale(X)

    # framestack
    if num_stack != 1:
        X, Y = create_framestack(X, Y, f_int, f_int2)

    # Data Augmentation - adding noise
    X, Y = add_noise(X, Y)

    # Training
    model.fit_batch(feed_dicts={network:X, labels:Y})
    train_acc, train_loss = model.session.run([acc, cost], feed_dict={network:X, labels:Y})
        
    train_summary = model.session.run(merged, feed_dict={network:X, labels:Y})
    writer2.add_summary(train_summary, i)
          
    if i%100 == 0:
        # get validation batch
        tx, ty = batch_get(val_name, 600)
        
        # feature scale validation data
        tx = feature_scale(tx)
        
        # Create validation framestack
        if num_stack != 1:
            tx, ty = create_framestack(tx, ty, f_int, f_int2)
        
        assert(len(ty) == tx.shape[0]),'data and label shapes do not match'
        
        # Get validation accuracy and error rate
        val_acc, val_loss, summary = model.session.run([acc, cost, merged], 
                                                   feed_dict={network:tx, labels:ty})
        writer.add_summary(summary, i)


# Save model and acc/error curves
model.save(m_save+modelswitch[model_num].__name__)
