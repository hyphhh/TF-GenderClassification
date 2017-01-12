import tensorflow as tf
import time
from utils import *


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
        list_path='data/train_list.txt',
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
