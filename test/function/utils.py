from PIL import Image
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters as ft
from scipy.signal import find_peaks
import sys
import cv2
import os
from pandas import DataFrame
import shutil

sys.path.append('../')

width_threshold_short = 5
H = 32
W = 44

def smooth(x,window_len=3,window='hamming'):
    w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),x,mode='same')
    return y

def cut_word_image(index_boundary, batch_img, batch_width, batch_idx, name, intermediate_show):
    # cut the word image
    cutted_char_images = []
    word_end = []
    for i in range(batch_img.shape[0]):
        boundary = index_boundary[i]
        width = batch_width[i]
        chars = np.split(batch_img[i], boundary, axis=1)[:-1]
        ends = np.zeros([len(chars)], dtype=int)
        ends[-1] = 1
        cutted_char_images.extend(chars)
        word_end.extend(list(ends))
        char_N = len(chars)
        # stack char images in list
        if intermediate_show:
            b_img = batch_img[i]
            b_img[:, boundary] += 0.5
            b_result = Image.fromarray(((b_img[:,:boundary[-1],0] - np.min(b_img))/(np.max(b_img) - np.min(b_img)) * 255.0).astype(np.uint8))
            b_result.save('word_boundary/' + name + '/out' + str(batch_idx) + '_' + str(i) + '.bmp')

    return cutted_char_images, word_end



def boundary_candidate_maker(batch_boundary, batch_width, batch_size, lower_probability = 0.4):
    batch_mask = np.indices(batch_boundary.shape)[1] < batch_width
    batch_boundary = batch_mask * batch_boundary
    boundary_candidate = np.zeros(batch_boundary.shape)
    for i in range(batch_boundary.shape[0]):
        boundary_candidate[i, find_peaks(batch_boundary[i], distance=width_threshold_short, height=lower_probability)[0]] = 1
    return boundary_candidate


def boundary_action_maker(batch_boundary, batch_width, batch_size, thr):
    # process for each word
    real_index_boundary = []
    real_boundary = np.zeros(batch_boundary.shape)
    for i in range(batch_size):
        real_boundary[i] = batch_boundary[i] > thr
        index_boundary = np.where(real_boundary[i])[0]
        # checkout whether boundaries are too close
        if 0 in index_boundary:
            index_boundary = np.delete(index_boundary, 0)

        if index_boundary.shape[0] == 0 or np.max(index_boundary) < batch_width[i] - 3:
            index_boundary = np.concatenate((index_boundary, np.reshape(batch_width[i] - 1, 1)))
        real_index_boundary.append(index_boundary)
    return real_index_boundary, real_boundary


def char_image_preprocessing(char_images, file_name, batch_index):
    result = np.ones([len(char_images), H, W, 1]) * 0.5
    for index in range(len(char_images)):
        img = char_images[index]
        # binarization, inverse
        img = img + 0.5
        if np.min(img) < np.max(img):
            bin_th = ft.threshold_otsu(img)
        else:
            bin_th = 0.5
        img = img > bin_th
        img = img - 0.5
        # padding to white background
        rate = min(H / img.shape[0], W / img.shape[1])
        img = cv2.resize(img, (int(rate * img.shape[1]), int(rate * img.shape[0])))
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])
        result[index, int((H - img.shape[0]) / 2):int((H - img.shape[0]) / 2) + img.shape[0],
        int((W - img.shape[1]) / 2):int((W - img.shape[1]) / 2) + img.shape[1], :] = img

    return result


def image_switching(input_batch, sess, lan, char_image, language):
    if len(language) > 1:
        feed_dict_switch = {char_image: input_batch}
        lang = sess.run(lan, feed_dict=feed_dict_switch)
        lan_idx = np.argmax(lang, axis=1)

        # default language is Korean
        kor_idx = np.where(lan_idx == 0)[0]

        if 'spe' in language:
            spe_idx = np.where(lan_idx == 1)[0]
        else:
            kor_idx = np.hstack((kor_idx, np.where(lan_idx == 1)[0]))
            spe_idx = np.array([], dtype=np.int64)

        if 'eng' in language:
            eng_idx = np.where(lan_idx == 2)[0]
        else:
            kor_idx = np.hstack((kor_idx, np.where(lan_idx == 2)[0]))
            eng_idx = np.array([], dtype=np.int64)

        if 'chi' in language:
            chi_idx = np.where(lan_idx == 3)[0]
        else:
            kor_idx = np.hstack((kor_idx, np.where(lan_idx == 3)[0]))
            chi_idx = np.array([], dtype=np.int64)

    else:
        # default language is Korean
        kor_idx = np.arange(input_batch.shape[0])
        eng_idx = np.array([], dtype=np.int64)
        spe_idx = np.array([], dtype=np.int64)
        chi_idx = np.array([], dtype=np.int64)

    return kor_idx, eng_idx, spe_idx, chi_idx


def image_recognizing(input_batch, idx, language, hypotheses, placeholders, sess, results, char_num, doc_num):
    if language == 'kor':
        lan = np.ones([idx.shape[0], 1], dtype=int) * 0
        step_load = hypotheses['kor']
        image_pl = placeholders['kor']
    elif language == 'spe':
        lan = np.ones([idx.shape[0], 1], dtype=int) * 1
        step_load = [hypotheses['spe'], hypotheses['spe'], hypotheses['spe']]
        image_pl = placeholders['spe']
    elif language == 'eng':
        lan = np.ones([idx.shape[0], 1], dtype=int) * 2
        step_load = [hypotheses['eng'], hypotheses['eng'], hypotheses['eng']]
        image_pl = placeholders['eng']
    elif language == 'chi':
        lan = np.ones([idx.shape[0], 1], dtype=int) * 3
        step_load = [hypotheses['chi'], hypotheses['chi'], hypotheses['chi']]
        image_pl = placeholders['chi']
    if bool(idx.shape[0] > 0):
        lan_batch = input_batch[idx]

        # recognizer
        feed_dict = {image_pl: lan_batch}
        step_vals = sess.run(step_load, feed_dict=feed_dict)
        cho_result = np.argmax(step_vals[0], 1)
        jung_result = np.argmax(step_vals[1], 1)
        jong_result = np.argmax(step_vals[2], 1)

        results[0] = np.append(results[0], cho_result)
        results[1] = np.append(results[1], jung_result)
        results[2] = np.append(results[2], jong_result)
        results[3] = np.append(results[3], idx)
        results[4] = np.append(results[4], lan)

    return results, char_num

def post_processing(char_cho_results, char_jung_results, char_jong_results, char_lan, char_word_end):
    char_total_N = len(char_cho_results)
    cho_results = []
    jung_results = []
    jong_results = []
    lan_results = []

    tess_idx = []


    temp_cho_results = []
    temp_jung_results = []
    temp_jong_results = []
    temp_lan_results = []

    word_idx = 0

    for char_index in range(char_total_N):
        temp_cho_results.append(char_cho_results[char_index])
        temp_jung_results.append(char_jung_results[char_index])
        temp_jong_results.append(char_jong_results[char_index])
        temp_lan_results.append(char_lan[char_index])
        if char_word_end[char_index] == 1:
            cho_results.append(temp_cho_results)
            jung_results.append(temp_jung_results)
            jong_results.append(temp_jong_results)
            lan_results.append(temp_lan_results)

            eng_idx = np.where(np.array(temp_lan_results) == 2)[0]

            temp = 0
            check = 0
            for i in range(len(eng_idx) - 1):
                if eng_idx[i] + 1 == eng_idx[i+1]:
                    temp+=1
                else:
                    check = max(temp+1, check)
                    temp = 0

            if check > int(len(temp_lan_results)*1/4):
                tess_idx.append(word_idx)
            temp_cho_results = []
            temp_jung_results = []
            temp_jong_results = []
            temp_lan_results = []

            word_idx += 1

    return cho_results, jung_results, jong_results, lan_results, tess_idx
