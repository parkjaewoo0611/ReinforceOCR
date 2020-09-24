import os
import numpy as np
import editdistance
import math

from function.data_loader import *
from function.utils import *
from function.model import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 8, """Mini-batch size""")
tf.app.flags.DEFINE_integer('epoch', 20, """Number of optimization steps to run""")
tf.app.flags.DEFINE_string('gpu', '0', """Device for training graph placement""")
tf.app.flags.DEFINE_float('l2_lambda', 0.00001, """mse_param""")
tf.app.flags.DEFINE_float('lr', 0.0001, """lr""")

tf.app.flags.DEFINE_string('segmenter_pretrained_model', './weights/Projection', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('switch_model', './weights/Switch', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_kor_model', './weights/Recognizer/kor', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_eng_model', './weights/Recognizer/eng', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_spe_model', './weights/Recognizer/spe', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_chi_model', './weights/Recognizer/chi', """Directory for model checkpoints""")

tf.logging.set_verbosity(tf.logging.INFO)

def main(argv=None):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
    lr = FLAGS.lr
    model_path = 'weights/Reinforce'
    lang_list = ['kor', 'eng', 'spe', 'chi']
    data_path = os.path.join('train')

    N_action = 10

    # number of training samples
    total_N = tfrecord_reader.N_of_tfrecord(data_path)

    # get training data
    get_image, get_cho, get_jung, get_jong, get_width, get_length, get_projection, get_language = tfrecord_reader.build_random_batch(data_path, FLAGS.batch_size)
    get_width = tf.squeeze(get_width)
    get_image = tf.reshape(get_image, [tf.shape(get_image)[0], tf.shape(get_image)[1], tf.shape(get_image)[2], 1])


    # global_step
    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign_add(global_step, 1, name='increment_global_step')



    # placeholders to update segmenter
    train_pl = tf.placeholder(tf.bool)
    word_image_pl = tf.placeholder(dtype = tf.float32, shape=[None, 32, None, 1])
    word_width_pl = tf.placeholder(dtype = tf.int32, shape=[None])
    word_label_pl = tf.placeholder(dtype = tf.float32, shape=[None, None])
    word_language_pl = tf.placeholder(dtype = tf.int32, shape=[None, None])
    word_length_pl = tf.placeholder(dtype = tf.int32, shape=[None, 1])
    A_dot_pl = tf.placeholder(dtype = tf.float32, shape=[None, None])
    B_dot_pl = tf.placeholder(dtype = tf.float32, shape=[None, None])
    projection_pl =  tf.placeholder(dtype = tf.float32, shape=[None, None])

    # Cutter
    logit = Segmenter(word_image_pl, word_width_pl, train_pl)
    Cut_restore = get_init_trained('Cutter')
    logit = tf.sigmoid(logit)

    # update cutter
    variables = tf.global_variables()
    Cut_vars = [var for var in variables if 'Cutter' in var.name]

    # learning rate
    learning_rate = tf.train.exponential_decay(lr,
                                               global_step,
                                               int(total_N / FLAGS.batch_size * FLAGS.epoch  / 3),
                                               0.1,
                                               staircase=False,
                                               name='learning_rate')

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        # L2 regularization
        l2_reg = l2_regularization('Cutter')

        # Boundary loss
        adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
        pol_grad = policy_gradient(l2_reg, logit, adam, A_dot_pl, B_dot_pl, Cut_vars, learning_rate, FLAGS.l2_lambda)
        train_op = adam.apply_gradients(pol_grad)



    # Recognizer & Switch to get reward
    char_image_pl = tf.placeholder(dtype = tf.float32, shape=[None, 32, 44, 1])

    # Switch
    swi = Switch(char_image_pl, False)
    Swi_restore = get_init_trained('Switch')

    #Recognizer
    rec_cho, rec_jung, rec_jong = Recognizer_kor(char_image_pl, False)
    rec_eng = Recognizer_eng(char_image_pl, False)
    rec_spe = Recognizer_spe(char_image_pl, False)
    rec_chi = Recognizer_chi(char_image_pl, False)
    Rec_kor_restore = get_init_trained('Recognizer_kor')
    Rec_eng_restore = get_init_trained('Recognizer_eng')
    Rec_spe_restore = get_init_trained('Recognizer_spe')
    Rec_chi_restore = get_init_trained('Recognizer_chi')


    # tensorboard summary
    s0 = tf.summary.scalar('learning_rate', learning_rate)
    s1 = tf.summary.scalar('l2_reg_loss', l2_reg)
    r_boundary_pl = tf.placeholder(tf.float32)
    s2 = tf.summary.scalar('reward_boundary', r_boundary_pl)
    s3 = tf.summary.image('image', word_image_pl)
    raw_logit = tf.expand_dims(tf.expand_dims(logit, 1), 3)
    bou_logit = tf.expand_dims(tf.expand_dims(word_label_pl, 1), 3)
    pro_tiled = tf.expand_dims(tf.expand_dims(projection_pl, 1), 3)
    pro_tiled = tf.cast(pro_tiled, tf.float32)
    raw_img = word_image_pl + tf.tile(raw_logit, [1, tf.shape(word_image_pl)[1], 1, 1])
    bou_img = word_image_pl + tf.tile(bou_logit, [1, tf.shape(word_image_pl)[1], 1, 1])
    pro_img = word_image_pl + tf.tile(pro_tiled, [1, tf.shape(word_image_pl)[1], 1, 1])
    s4 = tf.summary.image('raw_img', raw_img)
    s5 = tf.summary.image('boundary_img', bou_img)
    s6 = tf.summary.image('projection_img', pro_img)

    summary_op = tf.summary.merge([s0, s1, s2, s3, s4, s5, s6])

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver(Cut_vars, max_to_keep=1)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth =True))) as sess:
        sess.run(init_op)

        Cut_restore(sess, get_checkpoint(FLAGS.segmenter_pretrained_model))
        Swi_restore(sess, get_checkpoint(FLAGS.switch_model))
        Rec_kor_restore(sess, get_checkpoint(FLAGS.recog_kor_model))
        Rec_eng_restore(sess, get_checkpoint(FLAGS.recog_eng_model))
        Rec_spe_restore(sess, get_checkpoint(FLAGS.recog_spe_model))
        Rec_chi_restore(sess, get_checkpoint(FLAGS.recog_chi_model))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord=coord)
        train_writer = tf.summary.FileWriter(model_path)

        valid_r_prime = -math.inf
        step = 0
        epoch_step = int(total_N/FLAGS.batch_size)

        for e in range(0, FLAGS.epoch):
            lower_probability = 0.4 - 0.3 * e / FLAGS.epoch
            for index in range(0, epoch_step):
                # load data
                [image,\
                 width, \
                 cho,\
                 jung,\
                 jong,\
                 length,\
                 projection,\
                 language] = sess.run([get_image, \
                                       get_width,\
                                       get_cho, \
                                       get_jung, \
                                       get_jong, \
                                       get_length, \
                                       get_projection, \
                                       get_language])

                cho  = [np.trim_zeros(word) for word in cho.tolist()]
                jung = [np.trim_zeros(word) for word in jung.tolist()]
                jong = [np.trim_zeros(word) for word in jong.tolist()]

                # get boundary from cutter
                feed_dict = {word_image_pl : image,
                             word_width_pl : width,
                             train_pl : True}
                word_boundary = sess.run(logit, feed_dict=feed_dict)


                # select candidate boundary and scale to probability
                upper_probability = 0.99
                boundary_candidate = boundary_candidate_maker(word_boundary, width, lower_probability)
                word_boundary = word_boundary * boundary_candidate
                word_boundary = np.clip(word_boundary, None, upper_probability)

                # for a_j
                word_action = np.zeros([word_boundary.shape[0], word_boundary.shape[1], N_action])
                word_residu = np.zeros([word_boundary.shape[0], word_boundary.shape[1], N_action])

                # for p_a_j, r_a_j
                p_a = np.ones([FLAGS.batch_size, N_action])
                r_a = np.zeros([FLAGS.batch_size, N_action])


                #sampling actions
                word_action = boundary_action_maker(word_boundary, width, N_action=N_action, sampling = True)
                word_residu = boundary_candidate[...,np.newaxis] - word_action

                # cut word image with action
                char_images, batch_ends = cut_word_image(word_action, image)
                input_batch = char_image_preprocessing(char_images)
                total_imgs = np.concatenate(input_batch, 0)


                # Switching
                kor_idx, eng_idx, spe_idx, chi_idx = image_switching(total_imgs, sess, swi, char_image_pl, lang_list)
                action_length = [len(input_batch[i]) for i in range(N_action)]
                total_ends = np.array(sum(batch_ends, []))


                # Recognizing
                results = [np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty(0, int)]
                hypotheses = {}
                placeholders = {}
                if 'kor' in lang_list:
                    hypotheses['kor'] = [rec_cho, rec_jung, rec_jong]
                    placeholders['kor'] = char_image_pl
                    results = image_recognizing(total_imgs, kor_idx, 'kor',
                                                hypotheses, placeholders, sess, results)
                if 'eng' in lang_list:
                    hypotheses['eng'] = rec_eng
                    placeholders['eng'] = char_image_pl
                    results = image_recognizing(total_imgs, eng_idx, 'eng',
                                                hypotheses, placeholders, sess, results)
                if 'spe' in lang_list:
                    hypotheses['spe'] = rec_spe
                    placeholders['spe'] = char_image_pl
                    results = image_recognizing(total_imgs, spe_idx, 'spe',
                                                hypotheses, placeholders, sess, results)
                if 'chi' in lang_list:
                    hypotheses['chi'] = rec_chi
                    placeholders['chi'] = char_image_pl
                    results = image_recognizing(total_imgs, chi_idx, 'chi',
                                                hypotheses, placeholders, sess, results)

                idx_sort = np.argsort(results[3])
                results[0] = results[0][idx_sort]
                results[1] = results[1][idx_sort]
                results[2] = results[2][idx_sort]
                results[3] = np.arange(results[0].shape[0])
                results[4] = results[4][idx_sort]

                total_cho, total_jung, total_jong, total_lan = post_processing(results[0],
                                                                               results[1],
                                                                               results[2],
                                                                               results[4],
                                                                               total_ends)


                cho_results = []
                jung_results = []
                jong_results = []
                lan_results = []
                for j in range(0, N_action * FLAGS.batch_size, FLAGS.batch_size):
                    cho_results.append(total_cho[j  : j+FLAGS.batch_size])
                    jung_results.append(total_jung[j: j+FLAGS.batch_size])
                    jong_results.append(total_jong[j: j+FLAGS.batch_size])
                    lan_results.append(total_lan[j  : j+FLAGS.batch_size])

                for j in range(N_action):
                    for i in range(FLAGS.batch_size):
                        r_a[i, j] += 1 - editdistance.eval(cho_results[j][i],  cho[i])   / max(len(cho[i]),  len(cho_results[j][i]))
                        r_a[i, j] += 1 - editdistance.eval(jung_results[j][i], jung[i])  / max(len(jung[i]), len(jung_results[j][i]))
                        r_a[i, j] += 1 - editdistance.eval(jong_results[j][i], jong[i])  / max(len(jong[i]), len(jong_results[j][i]))
                        r_a[i, j] = r_a[i, j] / 3

                        p_a_1 = word_boundary[i]* word_action[i,:,j]
                        p_a_1 = np.prod(p_a_1[p_a_1 > 0])
                        p_a_2 = (1-word_boundary[i]) * word_residu[i,:,j]
                        p_a_2 = np.prod(p_a_2[p_a_2 > 0])
                        p_a[i, j] = p_a_1 * p_a_2


########################### projection profile to get r_prime #########################################################################
                prime_boundary = word_boundary[...,np.newaxis]
                prime_boundary = boundary_action_maker(prime_boundary, width, N_action=N_action, sampling = False)
                char_images, batch_ends = cut_word_image(prime_boundary, image)
                input_batch = char_image_preprocessing(char_images)

                # Switching
                total_imgs = input_batch[0]
                # Switching
                kor_idx, eng_idx, spe_idx, chi_idx = image_switching(total_imgs, sess, swi, char_image_pl, lang_list)
                total_ends = np.array(sum(batch_ends, []))

                # Recognizing
                results = [np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty(0, int), np.empty(0, int)]
                hypotheses = {}
                placeholders = {}
                if 'kor' in lang_list:
                    hypotheses['kor'] = [rec_cho, rec_jung, rec_jong]
                    placeholders['kor'] = char_image_pl
                    results = image_recognizing(total_imgs, kor_idx, 'kor',
                                                hypotheses, placeholders, sess, results)
                if 'eng' in lang_list:
                    hypotheses['eng'] = rec_eng
                    placeholders['eng'] = char_image_pl
                    results = image_recognizing(total_imgs, eng_idx, 'eng',
                                                hypotheses, placeholders, sess, results)
                if 'spe' in lang_list:
                    hypotheses['spe'] = rec_spe
                    placeholders['spe'] = char_image_pl
                    results = image_recognizing(total_imgs, spe_idx, 'spe',
                                                hypotheses, placeholders, sess, results)
                if 'chi' in lang_list:
                    hypotheses['chi'] = rec_chi
                    placeholders['chi'] = char_image_pl
                    results = image_recognizing(total_imgs, chi_idx, 'chi',
                                                hypotheses, placeholders, sess, results)

                idx_sort = np.argsort(results[3])
                results[0] = results[0][idx_sort]
                results[1] = results[1][idx_sort]
                results[2] = results[2][idx_sort]
                results[3] = np.arange(results[0].shape[0])
                results[4] = results[4][idx_sort]

                total_cho, total_jung, total_jong, total_lan = post_processing(results[0],
                                                                               results[1],
                                                                               results[2],
                                                                               results[4],
                                                                               total_ends)



                cho_results = total_cho
                jung_results = total_jung
                jong_results = total_jong
                lan_results = total_lan
                r_prime = np.zeros(FLAGS.batch_size)
                for i in range(prime_boundary.shape[0]):
                    r_prime[i] += 1 - editdistance.eval(cho_results[i], cho[i])  / max(len(cho[i]), len(cho_results[i]))
                    r_prime[i] += 1 - editdistance.eval(jung_results[i], jung[i])/ max(len(jung[i]), len(jung_results[i]))
                    r_prime[i] += 1 - editdistance.eval(jong_results[i], jong[i])/ max(len(jong[i]), len(jong_results[i]))
                    r_prime[i] = r_prime[i]/3
#################################################################################################################################



####################### get reward - reward_prime value ########################################################################
                # r_a_j - r_pr
                r_a_sub_p = r_a - np.expand_dims(r_prime, 1)
                A_dot = np.zeros(word_boundary.shape)
                B_dot = np.zeros(word_boundary.shape)
                for i in range(FLAGS.batch_size):
                    for j in range(N_action):
                        A_dot[i] += p_a[i, j] * r_a_sub_p[i, j] * word_action[i,:,j]
                        B_dot[i] += p_a[i, j] * r_a_sub_p[i, j] * word_residu[i,:,j]
                prime_boundary = prime_boundary[...,0]
                feed_dict = {word_image_pl : image,
                             word_width_pl : width,
                             word_label_pl : prime_boundary,
                             word_language_pl : language,
                             word_length_pl:length,
                             A_dot_pl : A_dot,
                             B_dot_pl : B_dot,
                             train_pl: True}
                [_, lr_, _] = sess.run([train_op, learning_rate, increment_global_step], feed_dict=feed_dict)

                if step % 10 == 0:
                    r_prime = np.mean(r_prime)
                    print('epoch: %i/%i  step: %i/%i r_prime: %.5f  lr: %.5f' % (FLAGS.epoch, e, epoch_step, step%epoch_step, r_prime, lr_))
                    print('index_prime_boundary: ', np.where(prime_boundary[0])[0])

####################### get summary  ##############################################################33
                if step%100 == 0:
                    feed_dict = {word_image_pl : image,
                                 word_width_pl : width,
                                 word_label_pl : prime_boundary,
                                 word_language_pl : language,
                                 word_length_pl:length,
                                 projection_pl : projection,
                                 A_dot_pl : A_dot,
                                 B_dot_pl : B_dot,
                                 r_boundary_pl : r_prime,
                                 train_pl: True}
                    [summary] = sess.run([summary_op], feed_dict=feed_dict)
                    train_writer.add_summary(summary, step)
                    if valid_r_prime < r_prime :
                        valid_r_prime = r_prime
                        saver.save(sess, os.path.join(model_path, 'model'))
                step += 1
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
