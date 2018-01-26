def resnext_block2(incoming, nb_blocks, out_channels, cardinality,
                  downsample=False, downsample_strides=1, activation='relu',
                  batch_norm=True, weights_init='variance_scaling',
                  regularizer='L2', weight_decay=0.0001, bias=True,
                  bias_init='zeros', trainable=True, restore=True,
                  reuse=False, scope=None, name="ResNeXtBlock"):

    _, h, w, in_channels = incoming.get_shape().as_list()
    b = 50
    out = tf.zeros([b, h, w, out_channels])
    if downsample:
        downsample_strides = 2
        out = tf.zeros([b, np.ceil(h/2.), np.ceil(w/2.), out_channels])
        print(h/2)
        print(w/2)
    for i in range(nb_blocks):
        _, _, _, in_channels = incoming.get_shape().as_list()
        for j in range(cardinality):
            resnext = conv_2d(incoming, out_channels, 3,
                              downsample_strides, 'same',
                              'linear', bias, weights_init,
                              bias_init, regularizer, weight_decay,
                              trainable, restore)
            if batch_norm:
                resnext = batch_normalization(resnext, trainable=trainable)
            resnext = tflearn.activation(resnext, activation)


            resnext = conv_2d(resnext, out_channels, 3,
                              1, 'same',
                              'linear', bias, weights_init,
                              bias_init, regularizer, weight_decay,
                              trainable, restore)
            if batch_norm:
                resnext = batch_normalization(resnext, trainable=trainable)
            resnext = tflearn.activation(resnext, activation)

            out = out + resnext

        if in_channels != out_channels:
            ch = (out_channels - in_channels)//2
            incoming = tf.pad(incoming,
                            [[0, 0], [0, 0], [0, 0], [ch, ch]])
        if downsample_strides > 1:
            incoming = avg_pool_2d(incoming, downsample_strides)
        out = out + incoming
        incoming = out

    return out
