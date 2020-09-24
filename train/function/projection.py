import numpy as np
from skimage import filters as ft
from operator import itemgetter
from scipy import ndimage
def projection_profiling(words, words_len):
    """make word images from a textline image words:Bx32xWxC, words_len:BxW
    words: numpy word image file 
    words_len: each word image length
    startidx: for output word image name, indicate startidx, type: binary(0) or not(1),
    height: 64 or 32,
    mode: 0(adaptive threshold), 1(mean threshold), 2(otsu threshold), other(determined threshold value),
    gauss: 1(filter gaussian before binarization), 0(or not)
    """

    type=0
    mode=2
    height=32
    gauss=1
    fast=1

    W = np.max(words_len)
    boundaries = np.zeros([words.shape[0], W], np.int32)

    for index in range(words_len.shape[0]):
        h, w = 32, words_len[index]
        word = words[index][:,:,0]

        fimage = ndimage.gaussian_filter(word, 0.9) #0.3
        th = ft.threshold_otsu(fimage)
        bimage = fimage > th
        bimage = bimage * 255
        bimage = 255 - bimage
        #bimage = 255 - th[1]
        bimage = ndimage.binary_dilation(bimage).astype(int)
        bimage = bimage * 255
    
        ht = np.zeros(w)
        for i in range(w):
            ht[i] = sum(bimage[:, i]) / 255
    
        pre_x = 0
        x = 0
    
        i = 0
        non_spaces = [] # each element indicates single non-space area, [start_x_index, end_x_index, length of the next space]
        isfirst = 0
        count = 0
        while i < w:
            if ht[i] <= 2: # space
                count = 0 # reset count
                while i+count < w - 1 and ht[i+count] <= 2:
                    count += 1
                x = i  # i+1
                if not isfirst and i != 0:
                    isfirst = 1
                if isfirst:
                    if x+count == w-1:
                        non_spaces.append([pre_x, x, 0])
                    else:
                        non_spaces.append([pre_x, x, count])
    
                isfirst = 1
                pre_x = x+count
                if i+count+1 < w:
                    i += count # count+1
                else:
                    break
            else:
                i += 1
        if pre_x < w-1:
            non_spaces.append([pre_x, w-1, count])
    
        if len(non_spaces) == 1:
            thresh = non_spaces[-1][2]+1
        elif len(non_spaces) == 0:
            return boundaries
        else:
            tmp = sorted(non_spaces, key=itemgetter(2))
            spaces = []
            for element in tmp:
                spaces.append(element[2])
            black = []
            for element in non_spaces:
                black.append(element[1]-element[0])
            black = np.array(black)
            hist, binn = np.histogram(spaces, bins=max(spaces), range=(0, max(spaces)))
            bbin = binn[0:-1]
            bbin[-1] = binn[-1]
            lam = lambda x, y: x*y
            value = lam(hist, bbin)
            thresh = 0
            maxval = 0
            for i in range(1, len(bbin)-1):
                alpha = sum(hist[0:i])/sum(hist)
                beta = 1-alpha
                tempval = alpha*beta*(sum(value[0:i])/sum(hist[0:i])-sum(value[i:])/sum(hist[i:]))**2
                if tempval >= maxval:
                    maxval = tempval
                    thresh = bbin[i]
        pre_x = 0
        x = 0
        token = 1
        end_token = 0
        idx = 0
        real_idx = 0
        is_space = 0

        final_spaces = []
        for character in non_spaces:
            start = character[0]
            end = character[1]
            space = character[2]
            
            final_spaces.append(min(end+space//2, w-1))
    
        first = 1
        for character in non_spaces:
            if token:
                #pre_x = character[0]
                pre_x = 0 if first else end
                first = 0
            x = character[1]
            token = 0
            if character[2] >= thresh:
                idx += 1
                start = max(pre_x-2, 0)
                end = final_spaces[real_idx]
                boundaries[index,end] = 1
#                end = min(x, w-1)
                is_space = 1
                if type:
                    image = word[:, start:end]
                else:
                    image = word[:, start:end]
                    if gauss:
                        temp_fimage = ndimage.gaussian_filter(image, 0.7) #0.7
                    else:
                        temp_fimage = image
                    if mode == 0:
                        bin_thresh = ft.threshold_local(temp_fimage, height-1, offset=4)
                    elif mode == 1:
                        bin_thresh = ft.threshold_mean(temp_fimage)
                    elif mode == 2:
                        bin_thresh = ft.threshold_otsu(temp_fimage)
                    else:
                        bin_thresh = mode
                    image = temp_fimage > bin_thresh
                    image = image * 255
                token = 1
            real_idx += 1   

        anchor = 0
        bbimage = 255 - bimage
        if is_space == 0:
            end = 0
        for i in range(end+1, w):
            if np.min(bbimage[:, i]) != 255:
                anchor = i
                break
    
        if anchor:
            end_token = 1
    
        if end_token:
            start = max(pre_x - 2, 0)
            idx += 1
            if type:
                end = x
                boundaries[index, end] = 1
                image = word[:, start:end]
            else:
                end = x
                boundaries[index, end] = 1
                image = word[:, start:end]
                if gauss:
                    temp_fimage = ndimage.gaussian_filter(image, 0.7) #0.7
                else:
                    temp_fimage = image
                if mode == 0:
                    bin_thresh = ft.threshold_local(temp_fimage, height-1, offset=4)
                elif mode == 1:
                    bin_thresh = ft.threshold_mean(temp_fimage)
                elif mode == 2:
                    bin_thresh = ft.threshold_otsu(temp_fimage)
                else:
                    bin_thresh = mode
                image = temp_fimage > bin_thresh
                image = image * 255

    return boundaries


