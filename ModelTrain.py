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


m_save = 'Sheri_3frames5,15_GrayCropped_RightAllDrivers_'

if 'Color' in m_save:
    im_method = 0
    num_stack = 1
    channs = 3
elif '3frames5,15_GrayCropped' in m_save:
    im_method = 1
    num_stack = 3
    channs = 3
    stack_nums = [5, 15]
elif '1frame_GrayCropped' in m_save:
    im_method = 2
    num_stack = 1
    channs = 1

# prompt the user for which model they want to train from NetworkSwitch.py
print(modelswitch)
model_num = np.int32(raw_input('Which model do you want to train (0 - ' + str(len(modelswitch)-1) + ')?'))

# start tensorboard 
os.system('tensorboard --logdir=/tmp/tflearn_logs/ &')

# define useful variables
os.chdir('/home/TF_Rover/RoverData/Right2')
fnames = glob.glob('*.h5') # datasets to train on
epochs = 20001 # number of training iterations
batch_sz = 80  # training batch size
val_name = 'Run_218seconds_Michael_Sheri.h5' # Dataset to use for validation
num_classes = 4


def add_noise(x, y):
    x_aug = x + np.random.randn(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    x = np.concatenate((x, x_aug), 0)
    y = np.concatenate((y, y), 0)
    return x, y



def create_framestack(x, y, f_args):
    f_args.sort()
    X_ = []
    Y_ = []
    
    for ex_num in range(x.shape[0]-1, max(f_args), -1):
        xf = x[ex_num, ...]

        for i in range(len(f_args)):
            xf = np.concatenate((xf,
                                 x[ex_num-f_args[i], ...]),
                                 axis=2)

        X_.append(xf)
        Y_.append(y[ex_num, :])
        
    return np.asarray(X_), np.asarray(Y_)



def feature_scale(x):
    b, h, w, c = x.shape
    x = scale(x.reshape([b, -1]), 1)
    return x.reshape([b, h, w, c])


def batch_get(filename, batch_size):
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    y = np.int32(f['Y']) + 1
    rand = np.random.randint(max(stack_nums), X.shape[0], batch_size)
    Y = np.zeros([batch_size, num_classes])
    Y[np.arange(batch_size), y[rand]] = 1.0
    X = X[rand, 110:, :, :]
    
    if im_method in [1, 2]:
        X = np.mean(X, 3, keepdims=True) # grayscale and crop frames
        
    assert(X.shape[0] == Y.shape[0]), 'Data and labels different sizes'
    f.flush()
    f.close()
    return X, Y


# Validation set
print('Validation Dataset: %s'%(val_name))

# Create input layer and label placeholder for the network
labels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])
network = tf.placeholder(dtype=tf.float32, shape=[None, 130, 320, channs])


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

    # local feature Scaling
    X = feature_scale(X)

    # framestack
    if num_stack != 1:
        X, Y = create_framestack(X, Y, stack_nums)

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
            tx, ty = create_framestack(tx, ty, stack_nums)
        
        assert(ty.shape[0] == tx.shape[0]),'data and label shapes do not match'
        
        # Get validation accuracy and error rate
        val_acc, val_loss, summary = model.session.run([acc, cost, merged], 
                                                   feed_dict={network:tx, labels:ty})
        writer.add_summary(summary, i)


# Save model and acc/error curves
model.save(m_save+modelswitch[model_num].__name__)
