import os
import pandas as pd
import PIL
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import threading as th

__dir__ = Path(__file__).absolute().parent
data_dir_ex = __dir__ /'assets/BAD/10 examples'
data_dir_large_b = __dir__ /'assets/BAD/More Examples'
data_dir_large_g = __dir__ /'assets/GOOD/More Examples'
data_save = __dir__ /'data'
resize_img_to = (56, 100)

def read_file(data_dir):
  return [f for f in glob.glob(data_dir + '/' + '*.png')]

def convert_file_to_numpy(image_file):
    """
    Take image URL (local or remote), open and convert the image to thumbnail
    :param image_file:
    :return: array of pixales
    """
    img = Image.open(image_file)
    img.thumbnail(resize_img_to, Image.Resampling.LANCZOS)
    return np.asarray(img)

def show_image(pic, mode='RGB'):
    """
    Plot the image
    :param pic:
    :param mode:
    :return:
    """
    inverted_im = PIL.Image.fromarray(pic, mode = mode)
    inverted_im.show()

def create_ds(im, lst_small, lst_large, y_small, y_large,ind):
    arr = convert_file_to_numpy(im)
    temp = 0 if ind < 160 else 1
    if (arr.shape[0] < 56)|(arr.shape[2] != 4):
        lst_large.append(arr)
        y_large.append(temp)
    else:
        lst_small.append(arr)
        y_small.append(temp)


# READ AMD PROCESS DATA
files = read_file(data_dir_large_b) + read_file(data_dir_large_g)
th_lst = []
lst_small = list()
lst_large = list()
y_small = list()
y_large = list()

for ind, im in enumerate(files):
  th_lst.append(th.Thread(target = create_ds, args=(im, lst_small, lst_large,y_small, y_large, ind,), daemon=True))
  th_lst[ind].start()

for thrd in th_lst:
  thrd.join()


# STACK THE ARRAYS AND SAVE THE DATA SET
data = np.stack(lst_large)

# np.save(data_save+"/data.npy", data)
np.save(data_save+"/data_resized.npy", data)
np.save(data_save+"/y.npy", y_large)