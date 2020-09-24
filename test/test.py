import os
import tensorflow as tf
from glob import glob
import math
import time
import numpy as np
import shutil
import cv2
import sys

from function.model import *
from function.utils import *
from function.label_error_rate import *
import function.decoding as fcjc

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('segmenter_path', './weights/Segmenter/RL_with_eps_decay', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('model_name', 'LSTM', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('switch_path', './weights/Switch', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_kor_path', './weights/Recognizers/kor', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_eng_path', './weights/Recognizers/eng', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_spe_path', './weights/Recognizers/spe', """Directory for model checkpoints""")
tf.app.flags.DEFINE_string('recog_chi_path', './weights/Recognizers/chi', """Directory for model checkpoints""")
tf.app.flags.DEFINE_integer('batch_size', 256, """Eval batch size""")
tf.app.flags.DEFINE_string('gpu', '0', """Device for graph placement""")
tf.app.flags.DEFINE_string('test_path', './input', """directory where input word numpy files are in""")
tf.app.flags.DEFINE_string('result_path', './output', """directory where output text files are stored in""")
tf.app.flags.DEFINE_string('language', 'kor, eng, spe, chi', """language to ocr. default ocr model is english""")
tf.app.flags.DEFINE_boolean('intermediate_show', True, """saving intermediate results(character segmentation, language switch, recognizer inputs) or not""")
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Non-configurable parameters
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

H = 32
W = 44
boundary_threshold = 0.5
width_threshold_long = 45
count = 10

class word_data:
    def __init__(self, numpy_glob, numpy_idx):
        self.loaded_numpy = np.load(numpy_glob[numpy_idx])

        fname = numpy_glob[numpy_idx].split("/")
        self.file_name = fname[len(fname) - 1].split(".")[0]
        self.results_folder = FLAGS.result_path + '/' + self.file_name
        # folder create
        if FLAGS.intermediate_show:
            if not os.path.exists('./word_boundary'):
                os.mkdir('./word_boundary')
            if not os.path.exists('./word_boundary/' + self.file_name):
                os.mkdir('./word_boundary/' + self.file_name)
            else:
                shutil.rmtree('./word_boundary/' + self.file_name)
                os.mkdir('./word_boundary/' + self.file_name)

        self.total_N = len(self.loaded_numpy)
        self.total_batch_N = math.ceil(len(self.loaded_numpy) / FLAGS.batch_size)
        self.current_index = 0

        self.line_end = np.zeros([self.total_N, 1])
        multi_col_index = np.where(self.loaded_numpy[:,2] == '1')[0]
        line_end_index = np.where(self.loaded_numpy[:, 1] == 'E')[0]
        self.line_end[line_end_index] = 1
        self.line_end[multi_col_index] = 2  # multi-columned
        self.batch_size = FLAGS.batch_size

        self.img = self.loaded_numpy[:, 0]

    def batch_index_loader(self):
        if self.current_index + FLAGS.batch_size < self.total_N:
            batch_indexes = range(self.current_index, self.current_index + FLAGS.batch_size)
            self.current_index += FLAGS.batch_size
        else:
            batch_indexes = range(self.current_index, self.total_N)
            self.current_index = self.total_N
            self.batch_size = len(batch_indexes)
        return batch_indexes

    def load_batch(self):
        # batch width and line end
        batch_indexes = self.batch_index_loader()
        batch_img = self.img[batch_indexes]

        batch_width = np.round(np.array([s.shape[1] for s in list(batch_img)]) * 32 / batch_img[0].shape[0])
        batch_width = np.expand_dims(batch_width.astype(int), 1)
        batch_max_width = np.int32(np.max(batch_width))
        input_batch = np.zeros([self.batch_size, H, batch_max_width, 1])

        for img_idx in range(self.batch_size):
            img = batch_img[img_idx] / 255.0
            img = cv2.resize(img, (int(img.shape[1] * 32 / img.shape[0]), 32))
            padded_one = np.ones([img.shape[0], batch_max_width - img.shape[1]]) / 2
            padded_img = np.concatenate((img, padded_one), axis=1) - 0.5
            input_batch[img_idx] = np.expand_dims(padded_img, 2)

        return input_batch, batch_width, batch_max_width, self.batch_size


start_time = time.time()


def main(argv=None):
    os.chdir('./')
    if not os.path.exists(FLAGS.result_path):
        shutil.rmtree('./output')
        os.mkdir(FLAGS.result_path)

    language = FLAGS.language.replace(" ", "").split(',')
    is_train = False
    tf.reset_default_graph()
    with tf.Graph().as_default():
        # Cutter model
        image = tf.placeholder(tf.float32, shape=[None, 32, None, 1], name='image')
        wid = tf.placeholder(tf.int32, [None], name='width')
        model_name = FLAGS.model_name
        boundary_logit = LSTM_Cutter(image, wid, is_train)
        boundary_logit = tf.nn.sigmoid(boundary_logit)
        cutter_restore_model = get_init_trained('Cutter')

        # Switch model
        char_image = tf.placeholder(tf.float32, shape=[None, 32, None, 1])
        lan = Switch(char_image, is_train)
        lan = tf.nn.softmax(lan)
        switch_restore_model = get_init_trained('Switch')

        # Recognizer model
        if 'kor' in language:
            char_image_kor = tf.placeholder(tf.float32, shape=[None, 32, None, 1])
            cho_logit_kor, jung_logit_kor, jong_logit_kor = Recognizer_kor(char_image_kor, is_train)
            recog_kor_restore_model = get_init_trained('Recognizer_kor')

        if 'spe' in language:
            char_image_spe = tf.placeholder(tf.float32, shape=[None, 32, None, 1])
            char_logit_spe = Recognizer_spe(char_image_spe, is_train)
            recog_spe_restore_model = get_init_trained('Recognizer_spe')

        if 'eng' in language:
            char_image_eng = tf.placeholder(tf.float32, shape=[None, 32, None, 1])
            char_logit_eng = Recognizer_eng(char_image_eng, is_train)
            recog_eng_restore_model = get_init_trained('Recognizer_eng')

        if 'chi' in language:
            char_image_chi = tf.placeholder(tf.float32, shape=[None, 32, None, 1])
            char_logit_chi = Recognizer_chi(char_image_chi, is_train)
            recog_chi_restore_model = get_init_trained('Recognizer_chi')


        # Open Session
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            # Restore model
            cutter_restore_model(sess, get_checkpoint(FLAGS.segmenter_path))
            switch_restore_model(sess, get_checkpoint(FLAGS.switch_path))
            if 'kor' in language:
                recog_kor_restore_model(sess, get_checkpoint(FLAGS.recog_kor_path))
            if 'spe' in language:
                recog_spe_restore_model(sess, get_checkpoint(FLAGS.recog_spe_path))
            if 'eng' in language:
                recog_eng_restore_model(sess, get_checkpoint(FLAGS.recog_eng_path))
            if 'chi' in language:
                recog_chi_restore_model(sess, get_checkpoint(FLAGS.recog_chi_path))

            print('Start OCR')

            # list page files
            numpy_glob = sorted(glob(FLAGS.test_path + '/*.npy'))
            print(numpy_glob)
            for numpy_idx in range(len(numpy_glob)):
                char_num = 0
                tic_sec = time.time()
                data = word_data(numpy_glob, numpy_idx)

                char_word_end = []

                char_cho_results = []
                char_jung_results = []
                char_jong_results = []
                char_lan = []
                index_boundary = []
                start_idx = 0

                for batch_idx in range(data.total_batch_N):
                    # load batch from data
                    input_batch, batch_width_matrix, batch_max_width, batch_size = data.load_batch()

                    # Use Cutter to get boundary of characters in word image
                    feed_dict = {image: input_batch, wid: batch_width_matrix[:, 0]}
                    raw_batch_boundary = sess.run(boundary_logit, feed_dict=feed_dict)

                    boundary_candidate = boundary_candidate_maker(raw_batch_boundary, batch_width_matrix, batch_size)
                    can_batch_boundary = raw_batch_boundary * boundary_candidate
                    batch_index_boundary, batch_boundary = boundary_action_maker(can_batch_boundary, batch_width_matrix, batch_size, boundary_threshold)
                    index_boundary += batch_index_boundary

                    char_images, char_ends = cut_word_image(batch_index_boundary, input_batch, batch_width_matrix, batch_idx, data.file_name, FLAGS.intermediate_show)

                    # preprocessing : padding, binarization, resizing
                    char_input_batch = char_image_preprocessing(char_images, data.file_name, batch_idx)
                    kor_idx, eng_idx, spe_idx, chi_idx = image_switching(char_input_batch, sess, lan, char_image, language)


                    # Recognize character using recognizer of each language
                    results = [np.empty(0, int),
                               np.empty(0, int),
                               np.empty(0, int),
                               np.empty(0, int),
                               np.empty(0, int),
                               []]


                    hypotheses = {}
                    placeholders = {}
                    if 'kor' in language:
                        hypotheses['kor'] = [cho_logit_kor, jung_logit_kor, jong_logit_kor]
                        placeholders['kor'] = char_image_kor
                        results, char_num = image_recognizing(char_input_batch, kor_idx, 'kor',
                                                    hypotheses, placeholders, sess, results, char_num, numpy_idx)
                    if 'eng' in language:
                        hypotheses['eng'] = char_logit_eng
                        placeholders['eng'] = char_image_eng
                        results, char_num = image_recognizing(char_input_batch, eng_idx, 'eng',
                                                    hypotheses, placeholders, sess,results, char_num, numpy_idx)
                    if 'spe' in language:
                        hypotheses['spe'] = char_logit_spe
                        placeholders['spe'] = char_image_spe
                        results, char_num = image_recognizing(char_input_batch, spe_idx, 'spe',
                                                    hypotheses, placeholders, sess, results, char_num, numpy_idx)
                    if 'chi' in language:
                        hypotheses['chi'] = char_logit_chi
                        placeholders['chi'] = char_image_chi
                        results, char_num = image_recognizing(char_input_batch, chi_idx, 'chi',
                                                    hypotheses, placeholders, sess, results, char_num, numpy_idx)

                    idx_sort = np.argsort(results[3])
                    results[0] = results[0][idx_sort]
                    results[1] = results[1][idx_sort]
                    results[2] = results[2][idx_sort]
                    results[3] = np.arange(results[0].shape[0])
                    results[4] = results[4][idx_sort]
                    results[5].sort()


                    char_word_end.extend(char_ends)
                    char_lan += results[4].tolist()
                    char_cho_results += results[0].tolist()
                    char_jung_results += results[1].tolist()
                    char_jong_results += results[2].tolist()

                    # batch process end


                # post process document and save results
                cho_results, jung_results, jong_results, lan_results, eng_word_index = post_processing(char_cho_results,
                                                                                                       char_jung_results,
                                                                                                       char_jong_results,
                                                                                                       char_lan,
                                                                                                       char_word_end)
                # decode recognizer output with character dictionary
                fcjc.reconstruct_cjc(cho_results, jung_results, jong_results,
                                     data.line_end,
                                     data.results_folder,
                                     lan_results)
                toc_sec = time.time()
                print(numpy_idx + 1, 'th Image Elapsed Time =%.2f seconds' % (toc_sec - tic_sec))


            print('OCR Finished')
            end_time = time.time()
            minute, second = divmod(end_time - start_time, 60)
            second = int(second)
            minute = int(minute)
            print('********************************')
            print('{} minutes {} seconds elapsed'.format(minute, second))

            acc_check()

if __name__ == '__main__':
    tf.app.run(main=main)
