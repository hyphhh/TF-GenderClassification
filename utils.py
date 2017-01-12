import tensorflow as tf


def distorted_inputs(list_path, image_dir, re_size, crop_size, batch_size, is_color=False):

    channel = 3 if is_color else 1

    with open(list_path,'r') as f:
        lines = f.readlines()
    path_label = [ image_dir + line.strip() for line in lines ]

    sample_queue = tf.train.string_input_producer(path_label, shuffle=True)
    sample = sample_queue.dequeue()

    path, label = tf.decode_csv(
        sample,
        [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string)],
        field_delim=' ')
    label = tf.string_to_number(label, tf.int32)

    image = tf.image.decode_jpeg(tf.read_file(path))
    image = tf.image.resize_images(image, [re_size, re_size])
    image = tf.random_crop(image, [crop_size, crop_size, channel])
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=63)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_whitening(image)

    image.set_shape([crop_size, crop_size, channel])

    batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=batch_size*10,
        min_after_dequeue=batch_size*2)

    return batch


def inputs(list_path, image_dir, re_size, crop_size, batch_size, is_color=False):

    channel = 3 if is_color else 1

    with open(list_path,'r') as f:
        lines = f.readlines()
    path_label = [ image_dir + line.strip() for line in lines ]

    sample_queue = tf.train.string_input_producer(path_label, shuffle=False)
    sample = sample_queue.dequeue()

    path, label = tf.decode_csv(
        sample,
        [tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.string)],
        field_delim=' ')
    label = tf.string_to_number(label, tf.int32)

    image = tf.image.decode_jpeg(tf.read_file(path))
    image = tf.image.resize_images(image, [re_size, re_size])
    image = tf.image.resize_image_with_crop_or_pad(image, crop_size, crop_size)
    image = tf.image.per_image_whitening(image)

    image.set_shape([crop_size, crop_size, channel])

    batch = tf.train.batch(
        [image, label],
        batch_size=batch_size)

    return batch


def batch_norm(x, phase_train, scope='bn', affine=True):
    """
    Batch normalization on convolutional maps.
    from: https://stackoverflow.com/questions/33949786/how-could-i-
    use-batch-normalization-in-tensorflow
    Only modified to infer shape from input tensor x.
    Parameters
    ----------
    x
        Tensor, 4D BHWD input maps
    phase_train
        boolean tf.Variable, true indicates training phase
    scope
        string, variable scope
    affine
        whether to affine-transform outputs
    Return
    ------
    normed
        batch-normalized maps
    """

    with tf.variable_scope(scope):
        og_shape = x.get_shape().as_list()
        if len(og_shape) == 2:
            x = tf.reshape(x, [-1, 1, 1, og_shape[1]])
        shape = x.get_shape().as_list()
        beta = tf.Variable(tf.constant(0.0, shape=[shape[-1]]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[shape[-1]]),
                            name='gamma', trainable=affine)

        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        ema_apply_op = ema.apply([batch_mean, batch_var])
        ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

        def mean_var_with_update():
            """Summary
            Returns
            -------
            name : TYPE
                Description
            """
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(phase_train,
                                          mean_var_with_update,
                                          lambda: (ema_mean, ema_var))

        normed = tf.nn.batch_norm_with_global_normalization(
            x, mean, var, beta, gamma, 1e-3, affine)
        if len(og_shape) == 2:
            normed = tf.reshape(normed, [-1, og_shape[-1]])
    return normed


def conv2d(x, num_out,
        is_train,
        kernel_h=3,
        kernel_w=3,
        strides_h=1,
        strides_w=1,
        padding='VALID',
        activation=tf.nn.relu,
        scope='conv2d',
        reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        
        w = tf.get_variable(
            'weights', [kernel_h, kernel_w, x.get_shape()[-1], num_out],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        b = tf.get_variable(
            'biases', [num_out],
            initializer=tf.constant_initializer(0.1))
        conv = batch_norm(tf.nn.conv2d(x, w, strides=[1, strides_h, strides_w, 1], padding=padding) + b, is_train)
        conv = activation(conv)

        return conv


def max_pool(x, ksize_h=2, ksize_w=2, strides_h=2, strides_w=2, padding='VALID', name='pooling'):

    return tf.nn.max_pool(
        x, ksize=[1, ksize_h, ksize_w, 1],
        strides=[1, strides_h, strides_w, 1],
        padding=padding,
        name=name)


def fully_connected(x, num_out,
                    is_train,
                    activation=tf.nn.relu,
                    scope='fc',
                    reuse=None):

    shape = x.get_shape().as_list()

    with tf.variable_scope(scope, reuse=reuse):

        w = tf.get_variable(
            'weights', [shape[1], num_out],
            initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))
        b = tf.get_variable(
            'biases', [num_out],
            initializer=tf.constant_initializer(0.1))
        fc = tf.matmul(x, w) + b

        fc = activation(fc)

        if is_train==True:
        
            fc = tf.nn.dropout(fc, 0.5)
        

        return fc