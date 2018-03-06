import tensorflow as tf
import tflearn
from NetworkSwitch import *

run_meta = tf.RunMetadata()
with tf.Session(graph=tf.Graph()) as sess:
    net = tf.placeholder(dtype=tf.float32, shape=[1, 130, 320, 1])
    net = ResNet26(net)
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))

