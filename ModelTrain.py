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
from NetworkSwitch import *
from sklearn.preprocessing import scale
from scipy.misc import imshow, imresize
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model_name',
    type=str,
    required=True,
    help='What to save the model under')
parser.add_argument(
    '--use_pretrained',
    type=str,
    default='n',
    help='Do you want to use a pretrained model?')
parser.add_argument(
    '--dropout_prob',
    type=float,
    required=False,
    default=0.5,
    help='What dropout probability to use if needed')
parser.add_argument(
    '--training_iters',
    type=int,
    default=10000,
    help='How many iterations of gradient descent to do')

args = parser.parse_args()
m_save = args.model_name + '_'
pt = args.use_pretrained
dropout_keep_prob = args.dropout_prob
training_iterations=args.training_iters

if pt == 'n':
    pt = False
elif pt in ['y', 'Y']:
    pt = True
#m_save = 'Sheri_3frames5,15_GrayCropped_RightAllDrivers_'
#pt = False

if 'Color' in m_save:
    im_method = 0
    num_stack = 1
    channs = 3
elif '3frames5,15_GrayCropped' in m_save:
    im_method = 1
    num_stack = 3
    channs = 3
elif '1frame_GrayCropped' in m_save:
    im_method = 2
    num_stack = 1
    channs = 1

# prompt the user for which model they want to train from NetworkSwitch.py
print(modelswitch)
model_num = np.int32(raw_input('Which model do you want to train (0 - ' + str(len(modelswitch)-1) + ')?'))

# load a pretrained model
if pt:
    fileName = glob.glob('/home/TF_Rover/RoverData/*.index')
    fileName = fileName[0]
    network = input_data(shape=[None, 130, 320, channs])
    modelFind = fileName[fileName.find('_', 64, len(fileName))+1:-6]
    assert(modelFind == modelswitch[model_num].__name__), 'different models'
    net_out = globals()[modelFind](network, dropout_keep_prob)
    model = tflearn.DNN(network)
    model.load(fileName[:-6])


# start tensorboard 
os.system('tensorboard --logdir=/tmp/tflearn_logs/ &')

# define useful variables
os.chdir(os.path.join(os.getcwd(), 'RoverData/Right2'))
fnames = glob.glob('*.h5') # datasets to train on
batch_sz = 150  # training batch size
val_name = 'Run_218seconds_Michael_Sheri.h5' # Dataset to use for validation
num_classes = 4
stack_nums = [5, 15]
learn_rate = 3e-5


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


def random_crop(x, padlen=30):
    h, w = x.shape[1], x.shape[2]
    X = np.zeros(x.shape)
    x = np.pad(x, 
              ((0,0), 
              (padlen//2,padlen//2), 
              (padlen//2,padlen//2), 
              (0,0)), 'constant')
    
    for i in range(x.shape[0]):
        h_ind, w_ind = np.random.randint(0, padlen, 2)
        X[i,...] = x[i, h_ind:h_ind+h, w_ind:w_ind+w, :]

    return X


def feature_scale(x):
    b, h, w, c = x.shape
    x = scale(x.reshape([b, -1]), 1)
    return x.reshape([b, h, w, c])


def batch_get(filename, batch_size, channs, num_classes):
    f = h5py.File(filename, 'r')
    X = f['X']
    Y = f['Y']

    x = np.zeros([batch_size, 130, 320, channs])
    y = np.zeros([batch_size, num_classes])
    rand = np.random.randint(max(stack_nums), X.shape[0], batch_size)
    count = 0
    for r in rand:
        x[count,...] = X[r, 110:, ...]
        y[count, int(Y[r] + 1.0)] = 1.0
        count += 1
    
    if channs == 1:
        X = np.mean(X, 3, keepdims=True) # grayscale and crop frames
        
    assert(X.shape[0] == Y.shape[0]), 'Data and labels different sizes'
    f.flush()
    f.close()
    return x, y


# Validation set
print('Validation Dataset: %s'%(val_name))

# Create input layer and label placeholder for the network
labels = tf.placeholder(dtype=tf.float32, shape=[None, num_classes])

if not pt:
    network = tf.placeholder(dtype=tf.float32, shape=[None, 130, 320, channs])
    net_out = modelswitch[model_num](network, dropout_keep_prob)


# send the input placeholder to the specified network
acc = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(net_out, 1), tf.argmax(labels, 1))))
cost = categorical_crossentropy(net_out, labels) # crossentropy loss function

# Tensorboard summaries
tf.summary.scalar('Accuracy_', acc)
tf.summary.scalar('Loss_', cost)
merged = tf.summary.merge_all()


# gradient descent optimizer
opt = tf.train.AdamOptimizer(learning_rate=learn_rate)
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

for i in range(training_iterations):
    
    # pick random dataset for this epoch
    n = np.random.randint(1, len(fnames)-1, 1)
    filename = fnames[n[0]]
    
    # skip validation set if chosen
    if filename == val_name: 
        continue

    # load the chosen data file
    X, Y = batch_get(filename, batch_sz, channs, num_classes)

    # local feature Scaling
    X = feature_scale(X)

    # framestack
    if num_stack != 1:
        X, Y = create_framestack(X, Y, stack_nums)

    # random crop for augmentation
    X = random_crop(X)

    # Training
    model.fit_batch(feed_dicts={network:X, labels:Y})
    train_acc, train_loss = model.session.run([acc, cost], feed_dict={network:X, labels:Y})
        
    train_summary = model.session.run(merged, feed_dict={network:X, labels:Y})
    writer2.add_summary(train_summary, i)
          
    if i%100 == 0:
        # get validation batch
        tx, ty = batch_get(val_name, 600, channs, num_classes)
        
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
