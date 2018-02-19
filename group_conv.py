def grouped_conv_2d(incoming, groups, filter_size, strides=1,
                    padding='same', activation='linear', bias=False,
                    weights_init='uniform_scaling', bias_init='zeros',
                    regularizer=None, weight_decay=0.001, trainable=True,
                    restore=True, reuse=False, scope=None,
                    name="GroupedConv2D"):

    b,h,w,in_channels = utils.get_incoming_shape(incoming)
    nb_filter = in_channels / groups

    for i in range(groups):

        block = incoming[..., i*nb_filter:i*nb_filter+nb_filter]

        if i == 0:

            inference = conv_2d(block, nb_filter, filter_size, strides, padding, 
                             activation, bias, weights_init, bias_init, 
                             regularizer, weight_decay, trainable, restore, 
                             reuse)
        if i > 0:
            group_conv = conv_2d(block, nb_filter, filter_size, strides, padding, 
                                 activation, bias, weights_init, bias_init, 
                                 regularizer, weight_decay, trainable, restore, 
                                 reuse)
            
            inference = tflearn.layers.merge_ops.merge([inference, group_conv],
                                                       'concat', 3)
            
    return inference
