import editdistance
import sys
sys.path.append('./function')
from function.label_error_rate import *
from glob import glob

def accurate_rate_one_page(hypo_file, label_file, mode):
    """ input sequence = text strings
        calculate edit distance for a page, as a whole sequence """

    fid_hypo = open(hypo_file, 'r', encoding='UTF8')
    hypo_lines = fid_hypo.readlines()
    fid_hypo.close()

    fid_label = open(label_file, 'r', encoding='UTF8')
    label_lines = fid_label.readlines()
    fid_hypo.close()

    if label_lines[0][0] == '\ufeff':
        label_lines[0] = label_lines[0][1:]
    if hypo_lines[0][0] == '\ufeff':
        hypo_lines[0] = hypo_lines[0][1:]

    labels = ''
    hypos = ''
    if mode == 0: # regardless of spacing
        for line in label_lines:
            if len(line) == 1:
                continue
            line = line.rstrip('\n')
            temp = line.split(' ')
            for word in temp:
                labels += word
        for line in hypo_lines:
            if len(line) == 1:
                continue
            line = line.rstrip('\n')
            temp = line.split(' ')
            for word in temp:
                hypos = hypos + word
    else: # regard spacing
        for line in label_lines:
            if len(line) == 1:
                continue
            labels = labels + line[0:-1] + ' '
        for line in hypo_lines:
            if len(line) == 1:
                continue
            if line == ' \n':
                continue
            if line[-1] == '\n':
                line = line[0:-1]
            if line[-1] == ' ':
                line = line[0:-1]
            hypos = hypos + line + ' '

    if labels[-1] == ' ':
        labels = labels[0:-1]
    if hypos[-1] == ' ':
        hypos = hypos[0:-1]
    edit_dist = editdistance.eval(hypos, labels)
    edit_dist_rate = edit_dist/len(labels)

    return 1 - edit_dist_rate

def acc_check():
    out_list = glob('./output/*.txt')
    out_list = sorted(out_list)
    for i, name in enumerate(out_list):
        name = name.split('/')[-1]
        hypo_file = './output/' + name
        label_file = './label/'+ name
        ac = accurate_rate_one_page(hypo_file, label_file, 0)
        print(str(ac*100)+"%")
