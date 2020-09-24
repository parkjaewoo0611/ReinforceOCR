import numpy as np
from skimage import filters as ft
from scipy.signal import find_peaks
import sys
import cv2
import os
sys.path.append('../')

boundary_threshold = 0.5
second_boundary_threshold = 0.05
width_threshold_short = 5
width_threshold_long = 40



def smooth(x,window_len=3,window='hamming'):
    w=eval('np.'+window+'(window_len)')
    y=np.convolve(w/w.sum(),x,mode='same')
    return y


def cut_word_image(boundary, batch_img):
    N_action = boundary.shape[-1]
    batch_size = batch_img.shape[0]

    batch_imgs = []
    batch_ends = []

    for j in range(N_action):
        action_imgs = []
        action_ends = []

        for i in range(batch_size):
            index = np.where(boundary[i, :, j])[0]
            chars = np.split(batch_img[i], index, axis=1)[:-1]
            action_imgs.extend(chars)

            ends = np.zeros([len(chars)], dtype=int).tolist()
            ends[-1] = 1
            action_ends.extend(ends)

        batch_imgs.append(action_imgs)
        batch_ends.append(action_ends)

    return batch_imgs, batch_ends



def boundary_candidate_maker(batch_boundary, batch_width, lower_probability):
    batch_size = batch_boundary.shape[0]
    batch_mask = np.indices(batch_boundary.shape)[1] < batch_width[..., np.newaxis]
    batch_boundary = batch_mask * batch_boundary
    boundary_candidate = np.zeros(batch_boundary.shape)

    batch_boundary[:,0] = 0
    batch_boundary[(range(0,batch_size),batch_width - 1)] = 2
    for i in range(batch_size):
        boundary_candidate[i, find_peaks(batch_boundary[i], distance=5, height=lower_probability)[0]] += 1
    return boundary_candidate


def boundary_action_maker(batch_boundary, batch_width, N_action=10, sampling=True):
    thr = boundary_threshold
    batch_size = batch_boundary.shape[0]
    batch_boundary[(range(batch_size), batch_width - 1)] = 2
    if sampling:
        real_boundary = np.tile(batch_boundary[...,np.newaxis], [1, 1, N_action])
        real_boundary = (real_boundary - np.random.sample(real_boundary.shape) > 0).astype(int)
    else:
        real_boundary = (batch_boundary > thr).astype(int)
    return real_boundary

def char_image_preprocessing(batch_imgs):
    H = 32
    W = 44
    N_action = len(batch_imgs)

    batch_result = []
    for j in range(N_action):
        action_imgs = batch_imgs[j]
        action_thrs = [ft.threshold_otsu(img + 0.5) - 0.5 if not np.max(img)==np.min(img) else 0.5 for img in action_imgs]
        action_imgs = [(img > thr).astype(float) - 0.5 for img, thr in zip(action_imgs, action_thrs)]

        N_chars = len(action_imgs)
        action_result = np.ones([N_chars, H, W, 1]) * 0.5
        for i in range(N_chars):
            img = action_imgs[i][..., 0]
            rate = min(H / img.shape[0], W / img.shape[1])
            img = cv2.resize(img, (int(rate * img.shape[1]), int(rate * img.shape[0])))[..., np.newaxis]
            action_result[i, int((H - img.shape[0]) / 2) : int((H - img.shape[0]) / 2) + img.shape[0],
                             int((W - img.shape[1]) / 2) : int((W - img.shape[1]) / 2) + img.shape[1], :] = img
        batch_result.append(action_result)

    return batch_result


def image_switching(input_batch, sess, lan, char_image, language):
    if len(language) > 1:
        # determine language
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

def image_recognizing(input_batch, idx, language, hypotheses, placeholders, sess, results):
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

    return results


def post_processing(char_cho_results, char_jung_results, char_jong_results, char_lan, char_word_end):
    char_total_N = len(char_cho_results)
    cho_results = []
    jung_results = []
    jong_results = []
    lan_results = []

    temp_cho_results = []
    temp_jung_results = []
    temp_jong_results = []
    temp_lan_results = []
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
            temp_cho_results = []
            temp_jung_results = []
            temp_jong_results = []
            temp_lan_results = []
    return cho_results, jung_results, jong_results, lan_results


