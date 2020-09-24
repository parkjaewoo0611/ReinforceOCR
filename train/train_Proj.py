import os
import numpy as np
from function.data_loader import *
from function.model import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 32, """Mini-batch size""")
tf.app.flags.DEFINE_integer('epoch', 10, """Number of optimization steps to run""")
tf.app.flags.DEFINE_string('gpu', '0', """Device for training graph placement""")
tf.app.flags.DEFINE_float('lr', 0.001, """Directory for model checkpoints""")

tf.logging.set_verbosity(tf.logging.INFO)


def main(argv=None):
    os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    model_path = 'weights/Projection'
    data_path = 'train'

    global_step = tf.train.get_or_create_global_step()

    train_pl = tf.placeholder(tf.bool)

    image, _, _, _, width, _, proj, _ = tfrecord_reader.build_random_batch(data_path, FLAGS.batch_size)
    proj = tf.cast(proj, tf.float32)
    width = tf.squeeze(width)
    image = tf.reshape(image, [tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2], 1])

    # Cutter
    logit = Segmenter(image, width, train_pl)

    # loss
    cutter_los = sig_ce_loss(logit, proj)
    logit = tf.sigmoid(logit)
    Cut_restore = get_init_trained('Cutter')


    variables = tf.global_variables()
    Cut_vars = [var for var in variables if 'Cutter' in var.name]

    # learning rate
    total_N = tfrecord_reader.N_of_tfrecord(data_path)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        learning_rate = tf.train.exponential_decay(FLAGS.lr,
                                                   tf.train.get_global_step(),
                                                   int(total_N / FLAGS.batch_size * FLAGS.epoch / 3),
                                                   0.1,
                                                   staircase=False,
                                                   name='learning_rate')
        # Cutter loss
        Cutter_adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
        Cutter_train_op = tf.contrib.layers.optimize_loss(loss = cutter_los,
                                                          global_step = tf.train.get_global_step(),
                                                          learning_rate = learning_rate,
                                                          optimizer = Cutter_adam,
                                                          variables = Cut_vars)


    s0 = tf.summary.scalar('learning_rate', learning_rate)
    s1 = tf.summary.scalar('Cutter_loss', cutter_los)
    s2 = tf.summary.image('image', image)
    raw_logit = tf.expand_dims(tf.expand_dims(logit, 1), 3)
    pred_logit = tf.expand_dims(tf.expand_dims(tf.cast(tf.greater(logit, 0.5), tf.float32), 1), 3)
    proj_tiled = tf.expand_dims(tf.expand_dims(proj, 1), 3)
    raw_img = image + tf.tile(raw_logit, [1, tf.shape(image)[1], 1, 1])
    pred_img = image + tf.tile(pred_logit, [1, tf.shape(image)[1], 1, 1])
    proj_img = image + tf.tile(proj_tiled, [1, tf.shape(image)[1], 1, 1])
    s3 = tf.summary.image('raw_img', raw_img)
    s4 = tf.summary.image('prediction_img', pred_img)
    s5 = tf.summary.image('projection_img', proj_img)

    summary_op = tf.summary.merge([s0, s1, s2, s3, s4, s5])

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    Cut_saver = tf.train.Saver(Cut_vars, max_to_keep=1)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))) as sess:
        sess.run(init_op)

        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord=coord)
        train_writer = tf.summary.FileWriter(model_path)

        valid_loss = np.inf
        step = 0
        cur_epoch = 0
        epoch_step = int(total_N/FLAGS.batch_size)

        for e in range(cur_epoch, FLAGS.epoch):
            for index in range(0, epoch_step):
                feed_dict = {train_pl:True}
                if step%10 == 0:
                    [_, Cutter_loss_, lr_, logit_, width_, summary] = sess.run([Cutter_train_op, cutter_los, learning_rate, logit, width, summary_op], feed_dict=feed_dict)
                    print('epoch: %i/%i  step:%i/%i cutter_loss: %.5f lr: %.8f'\
                            % (FLAGS.epoch, e, epoch_step, step%epoch_step, Cutter_loss_, lr_))
                    train_writer.add_summary(summary, step)
                else:
                    sess.run(Cutter_train_op, feed_dict=feed_dict)

                if step % 100 ==  0:
                    feed_dict={train_pl:False}
                    [valid_loss_, valid_logit_, valid_width_] = sess.run([cutter_los, logit, width], feed_dict=feed_dict)
                    print('validation epoch: %i/%i  step:%i/%i cutter_loss: %.5f lr: %.8f'\
                            % (FLAGS.epoch, e, epoch_step, step%epoch_step, valid_loss_, lr_))

                    if valid_loss_ < valid_loss:
                        valid_loss = valid_loss_
                        Cut_saver.save(sess,
                                       os.path.join(model_path, 'model.ckpt'),
                                       global_step=step)

                step += 1
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
