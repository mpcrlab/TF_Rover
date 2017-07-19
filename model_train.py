# author = michael teti

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
from sklearn.preprocessing import scale


# prompt the user to choose the name of the saved model
print('What filename do you want to save this model as?')
m_save = raw_input('Rover Used_number of frames/stackinterval_other parameters  ')

# prompt the user for which model they want to train from NetworkSwitch.py
model_num = np.int32(raw_input('Which model do you want to train (0 - 10)?'))

# define useful variables
os.chdir('/home/TF_Rover/RoverData')
fnames = glob.glob('*.h5') # datasets to train on
epochs = 350 # number of epochs
batch_sz = 70  # training batch size
errors = []  # variable to store the validation losses
test_num = 600  # Number of validation examples
f_int = 2
f_int2 = 5
val_accuracy = [] # variable to store the validation accuracy
num_stack = 1
val_name = 'Run_2_both_lights_on.h5' # Dataset to use for validation
num_iters = 0.



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
tx = feature_scale(tx)

# Create validation framestack
#tx, ty = create_framestack(tx, ty, f_int, f_int2)
#assert(TY.shape[0] == TX.shape[0]),'data and label shapes do not match'


# Create input layer and label placeholder for the network
labels = tf.placeholder(dtype=tf.float32, shape=[None, 3])
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
opt = tf.train.AdamOptimizer(learning_rate=0.0001)
trainop = tflearn.TrainOp(loss=cost,
                         optimizer=opt,
                         metric=None,
                         batch_size=batch_sz)
model = Trainer(train_ops=trainop)

writer = tf.summary.FileWriter('/tmp/tflearn_logs/test'+m_save+modelswitch[model_num].__name__,
                               model.session.graph)
writer2 = tf.summary.FileWriter('/tmp/tflearn_logs/train'+m_save+modelswitch[model_num].__name__,
                               model.session.graph)

for i in range(epochs):
    
    # pick random dataset for this epoch
    n = np.random.randint(1, len(fnames)-1, 1)
    filename = fnames[n[0]]
    
    # skip validation set if chosen
    if filename == val_name: 
        print('skipping iteration')
        continue

    # load the chosen data file
    f = h5py.File(filename, 'r')
    X = np.asarray(f['X'])
    y = np.int32(f['Y']) + 1
    Y = np.zeros([y.shape[0], 3])
    rand = np.random.randint(0, X.shape[0], X.shape[0])
    Y[np.arange(Y.shape[0]), y] = 1.0 # create one-hot label vector
    X = np.mean(X[:, 110:, :, :], 3, keepdims=True) # grayscale and crop frames
    assert(X.shape[0] == Y.shape[0]), 'Data/label dimensions not equal'
    num_batches = np.int32(np.ceil(X.shape[0]/batch_sz))
    train_error=[]

    for j in range(num_batches):
        if j * batch_sz + batch_sz <= (X.shape[0]-1):
            x = X[j*batch_sz:j*batch_sz+batch_sz, :, :, :]
            y = Y[j*batch_sz:j*batch_sz+batch_sz, :]
        elif j*batch_sz + batch_sz >= (X.shape[0]-1) and X.shape[0]-(j*batch_sz + batch_sz)>=f_int2:
            x = X[j*batch_sz:X.shape[0], :, :, :]
            y = Y[j*batch_sz:X.shape[0], :]
        else:
            continue
   
        # local feature Scaling
        x = feature_scale(x)

        # framestack
        #x, y = create_framestack(x, y, f_int, f_int2)

        # Data Augmentation
        x, y = add_noise(x, y)

        # Training
        model.fit_batch(feed_dicts={network:x, labels:y})
        train_acc, train_loss = model.session.run([acc, cost], feed_dict={network:x, labels:y})
        #sys.stdout.write('Epoch %d; dataset %s; train_acc: %.2f; loss: %f  \r'%(
        #                            i+1, filename, train_acc, train_loss) )
        #sys.stdout.flush()
        num_iters += 1.
        
        if num_iters%20 == 0:
            train_summary = model.session.run(merged, feed_dict={network:x, labels:y})
            writer2.add_summary(train_summary, num_iters)
          
        elif num_iters%200 == 0:
            # Get validation accuracy and error rate
            val_acc, val_loss, summary = model.session.run([acc, cost, merged], 
                                                   feed_dict={network:tx, labels:ty})
            writer.add_summary(summary, num_iters)

            
    f.flush()
    f.close()

# Save model and acc/error curves
model.save(m_save+modelswitch[model_num].__name__)
