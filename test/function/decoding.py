import numpy as np
import os
import glob

# -*- coding:utf-8 -*-

def dictionary_loader(dictionary_file = 'dictionary_hangul.txt', length=2350, lan=0):
    f = open(dictionary_file,'r', encoding='UTF8')
    if lan == 0:
        dictionary = [['?', 0, 0, 0, 0]]
        for i in range(0, length+1):
            if i == 0:
                line = f.readline()
            else:
                line = f.readline()
                line = line.split()
                dictionary.append([line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4])])
    else:
        dictionary = [['?', 0]]
        for i in range(0, length+1):
            if i == 0:
                line = f.readline()
            else:
                line = f.readline()
                line = line.split()
                dictionary.append([line[0], int(line[1])])
    f.close()
    return dictionary


def reconstruction_hangul_normal(cho,jung,jong,dictionary):
    if cho==0 and jung==0 and jong==0:
        char = '?'
        char_label = 0
    else:
        w = 44032 + 28 * 21 * (cho-1) + 28 * (jung-1) + (jong-1)
        char = chr(int(w))
        char_idx = np.isin(dictionary[:, 0], char) * 1
        if (sum(char_idx) == 0):
            char_label = 0 #1
        else:
            char_label = np.argmax(char_idx)
    return char, char_label


def reconstruction_other_normal(char_,dictionary):
    char_idx = np.isin(dictionary[:,1], str(int(char_))) * 1
    if (sum(char_idx) == 0):
        char_label = 0 #1
    else:
        char_label = np.argmax(char_idx)
    char = dictionary[char_label,0]
    return char, char_label


def reconstruct_cjc(cho_results,jung_results,jong_results, enter_idx_mat, output_folder,lan_results):
    """ total_label_list
    Details including exception are presented in PPT
    :param cho_results:  ctc based cho results
    :param jung_results:  ctc based jung results
    :param jong_results:  ctc based jong results
    :param enter_idx_mat: line space index
    :param output_folder: output folder
    :param char_images: character images
    :return: outputs txt file in output folder
    """
    word = []
    label_list = []
    dictionary_kor = dictionary_loader('./function/dictionary_hangul.txt', 2350, 0)
    dictionary_spe = dictionary_loader('./function/dictionary_special.txt', 46, 1)
    dictionary_eng = dictionary_loader('./function/dictionary_english.txt', 52, 2)
    dictionary_chi = dictionary_loader('./function/dictionary_chinese.txt', 1817, 3)

    count = 0
    for i in range(len(cho_results)):
        cho_line = cho_results[i]
        jung_line = jung_results[i]
        jong_line = jong_results[i]

        number_label_cho = len(cho_line)
        number_label_jung = len(jung_line)
        number_label_jong = len(jong_line)

        temp = ''
        temp2 = ''
        length_array = [number_label_cho,number_label_jung,number_label_jong]
        max_cjj = np.max([length_array])
        dictionary_kor = np.asarray(dictionary_kor)
        dictionary_spe = np.asarray(dictionary_spe)
        dictionary_eng = np.asarray(dictionary_eng)
        dictionary_chi = np.asarray(dictionary_chi)
        for j in range(0, max_cjj):
            if lan_results[i][j] == 0:
                char, char_label = reconstruction_hangul_normal(cho_line[j], jung_line[j], jong_line[j], dictionary_kor)
            elif lan_results[i][j] == 1:
                char, char_label = reconstruction_other_normal(cho_line[j], dictionary_spe)
            elif lan_results[i][j] == 2:
                char, char_label = reconstruction_other_normal(cho_line[j], dictionary_eng)
            elif lan_results[i][j] == 3:
                char, char_label = reconstruction_other_normal(cho_line[j], dictionary_chi)
            temp = temp + char
            temp2 =  temp2 + str(char_label) + ' '
        temp = temp + ' '
        if enter_idx_mat[i] == 1:
            temp = temp + '\n'
            temp2 = temp2 + '\n'
        elif enter_idx_mat[i] == 2:
            temp = temp + '          '  # 10 spaces
            temp2 = temp2 + '          '
        word.append(temp)
        label_list.append(temp2)

    save_name = output_folder + '.txt'
    out = open(save_name, 'w', encoding='utf-8')
    for i in range(len(cho_results)):
        out.write(word[i])
    out.closed
