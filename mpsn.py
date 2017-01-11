import tensorflow as tf
import time


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


def cnn_model(image, is_train, reuse=None):

    conv1 = conv2d(
        image, num_out=64,
        is_train=is_train,
        kernel_h=7,
        kernel_w=7,
        strides_h=2,
        strides_w=2,
        scope='conv1',
        reuse=reuse)

    pool1 = max_pool(conv1, name='pool1')

    conv2 = conv2d(
        pool1, num_out=128,
        is_train=is_train,
        kernel_h=3,
        kernel_w=3,
        strides_h=2,
        strides_w=2,
        scope='conv2',
        reuse=reuse)

    pool2 = max_pool(conv2, name='pool2')

    conv3a = conv2d(
        pool2, num_out=256,
        is_train=is_train,
        scope='conv3a',
        reuse=reuse)

    conv3b = conv2d(
        conv3a, num_out=256,
        is_train=is_train,
        scope='conv3b',
        reuse=reuse)

    conv3c = conv2d(
        conv3b, num_out=256,
        is_train=is_train,
        scope='conv3c',
        reuse=reuse)

    pool3 = max_pool(conv3c, name='pool3')

    pool3_flat = tf.reshape(pool3, [-1, 4*4*256])
    # conv3c_flat = tf.reshape(conv3c, [-1, 8*8*160])

    fc1 = fully_connected(
        pool3_flat, num_out=2048,
        is_train=is_train,
        scope='fc1',
        reuse=reuse)

    fc2 = fully_connected(
        fc1, num_out=2048,
        is_train=is_train,
        scope='fc2',
        reuse=reuse)

    fc3 = fully_connected(
        fc2, num_out=2,
        is_train=False,
        activation=lambda x: x,
        scope='fc3',
        reuse=reuse)

    fc4 = fully_connected(
        fc2, num_out=512,
        is_train=False,
        activation=lambda x: x,
        scope='fc4',
        reuse=reuse)

    return fc3, fc4


def contrastive_loss(y, y_, y_p, y_p_):

    # same(1) or not same(0) label
    ybin_ = tf.cast(y_, tf.bool)
    ybin_p_ = tf.cast(y_p_, tf.bool)
    label = tf.cast(tf.logical_not(tf.logical_xor(ybin_, ybin_p_)), tf.float32)

    # distance
    d = tf.sqrt(tf.reduce_sum(tf.square(y - y_p),1))

    return tf.reduce_mean(label*tf.square(d) + (1-label)*tf.square(tf.maximum(0., 2-d)))


def process(coord, max_iter, snapshot_iter, snapshot_path, display, continuous_model=None):

    start_iter = 0

    if continuous_model:

        saver.restore(sess, continuous_model)
        s = continuous_model.split('_')
        start_iter = int(s[-1])

    try:
        for i in range(max_iter):

            # save model
            if i%snapshot_iter==0 and i!=0:

                print("saving model...")
                saver.save(sess, snapshot_path+'/model_iter_'+str(i+start_iter))

            # print training loss
            if i%display==0:

                train_loss, closs, closs_p, mloss, lr = sess.run(
                    [loss, cross_entropy, cross_entropy_p, metric_loss, learning_rate],
                    feed_dict={is_train: False})

                print("Step %d, %s, Phase: Train, lr: %f"%(i+start_iter, time.asctime(time.localtime(time.time())), lr))
                print("    train_loss: %f, closs1: %f, closs2: %f, mloss: %f"%(train_loss, closs, closs_p, mloss))

            # train
            sess.run(train_step, feed_dict={is_train: True})

        coord.request_stop()

    except Exception as e:

        coord.request_stop(e)

    finally:

        print('Done training.')

        saver.save(sess, snapshot_path+'/model_iter_'+str(i+start_iter))


if __name__ == '__main__':

    # input
    is_train = tf.placeholder(tf.bool)

    batch = distorted_inputs(
        list_path='data1/train_list.txt',
        image_dir='../adience_250/',
        re_size=250,
        crop_size=233,
        batch_size=32,
        is_color=True)

    batch_train = [batch[0][:16], batch[1][:16]]
    batch_train_p = [batch[0][16:32], batch[1][16:32]]

    # output and loss
    y, fcs = cnn_model(batch_train[0], is_train)
    y_p, fcs_p = cnn_model(batch_train_p[0], is_train, reuse=True)

    y_ = batch_train[1]
    y_p_ = batch_train_p[1]

    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cross_entropy_p = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_p, y_p_))
    metric_loss = contrastive_loss(fcs, y_, fcs_p, y_p_)

    loss = cross_entropy + cross_entropy_p + 0.1*metric_loss

    # train step
    global_step = tf.Variable(0, tf.int32)
    starter_learning_rate = 0.0001
    learning_rate = tf.train.exponential_decay(
        learning_rate=starter_learning_rate,
        global_step=global_step,
        decay_steps=2000,
        decay_rate=0.95,
        staircase=True,
        name=None)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # model saver
    saver = tf.train.Saver()
    saver.__init__(max_to_keep=50)

    # run
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    process(coord=coord,
        max_iter=100000,
        snapshot_iter=2000,
        snapshot_path='model_mpsn',
        display=100,
        continuous_model=None)

    coord.request_stop()

    coord.join(threads)

    sess.close()
