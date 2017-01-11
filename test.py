import tensorflow as tf
import mpsn


def test(model_path, batch_size, test_interval):

    # with tf.Graph().as_default(), tf.device('/cpu:0'):
    is_train = tf.placeholder(tf.bool)

    batch_test = mpsn.inputs(
        list_path='data1/test_list.txt',
        image_dir='../adience_250/',
        re_size=250,
        crop_size=233,
        batch_size=batch_size,
        is_color=True)

    y_ = batch_test[1]
    y, _ = mpsn.cnn_model(batch_test[0], is_train)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.cast(y_, tf.int64))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    with tf.device("/cpu:0"):
        sess.run(tf.initialize_all_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver.restore(sess, model_path)

        sum_acc=0
        for i in range(test_interval):
            print 'iter '+str(i)
            sum_acc += sess.run(accuracy, feed_dict={is_train: False})
        print sum_acc/test_interval

        coord.request_stop()
        coord.join(threads)

    sess.close()


if __name__ == '__main__':

    test(model_path='model_mpsn/model_iter_10000', batch_size=28, test_interval=103)
