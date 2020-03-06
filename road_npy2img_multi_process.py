import numpy as np
import os
import scipy.misc
import cv2
import cvbase as cvb

new_list = [0, 200, 204, 213, 209, 206, 207, 201, 203, 211, 208, 216, 217, 
            215, 218, 219, 210, 232, 214, 202, 220, 221, 222, 231, 224, 225, 
            226, 230, 228, 229, 233, 205, 212, 227, 223, 250, 249, 255]

def func(image_name):
    origin_npy = np.load('road05_tmp/' + image_name)
    # print(origin_npy.shape)
    tmp_npy = np.zeros(origin_npy.shape, dtype='uint8')
    final_npy = np.zeros((2710, 3384), dtype='uint8')
    for cnt in range(len(new_list)):
        tmp_npy = tmp_npy + (origin_npy == (cnt + 1)) * new_list[cnt]
    # tmp_npy = tmp_npy + (origin_npy == 1) * 0 # 255
    origin_npy = tmp_npy
    origin_npy = cv2.resize(origin_npy, (3384, 1010), interpolation=cv2.INTER_NEAREST).astype('uint8')
    # print(origin_npy.shape)
    final_npy[1700:, :] = origin_npy
    origin_npy = final_npy
    assert(origin_npy.shape == (2710, 3384))
    cv2.imwrite('road05/' + image_name.replace('.npy', '.png'), origin_npy) # .astype('uint8'))

home_directory = 'road05_tmp'
image_list = os.listdir(home_directory)
cvb.track_parallel_progress(func, image_list, 8)
