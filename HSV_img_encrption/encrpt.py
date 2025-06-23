import string

import cv2
import sys
import random
import copy
import time
from PIL import Image
# sys.path.append("utils")
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from key_generate import generate_keyb
from key_generate import generate_keys
from key_generate import generate_keyp
from key_generate import generate_keyv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from pyope.ope import OPE, ValueRange

BLOCK_SIZE = 36
block_size = 9

random_str1 = OPE.generate_key()
random_str2 = OPE.generate_key()
random_str3 = OPE.generate_key()
random_str4 = OPE.generate_key()
print(random_str1)
key = []
key.append(random_str1)
key.append(random_str2)
key.append(random_str3)
key.append(random_str4)
ope = []
for i in range(4):
    cipher = OPE(key[i], in_range=ValueRange(0, 255),
                                out_range=ValueRange(0, 255))
    ope.append(cipher)
cipt=ope[3].encrypt(222)
print(cipt)

def read_image(path):
    img = cv2.imread(path)
    return img




# 获取大块并将其随机排列

def get_image_big_block(img, block_no):
    h, w = img.shape[0], img.shape[1]
    start = block_no * BLOCK_SIZE

    row = start // w
    #    col_start = (block_no%2)*BLOCK_SIZE
    col_start = block_no * BLOCK_SIZE
    if col_start >= w:
        col_start = ((col_start % w) // BLOCK_SIZE) * 36
    if col_start == w:
        return
    col_end = (col_start + BLOCK_SIZE)

    pix = []

    for i in range(col_start, col_end):
        pix.append([img[row][i][0], img[row][i][1], img[row][i][2]])

    return pix

def get_image_small_block(bigbolock, block_no):
    # h, w = bigbolock.shape[0], bigbolock.shape[1]
    start = block_no * block_size

    end = start + block_size
    pix = []

    for i in range(start, end):
        pix.append([bigbolock[i][0], bigbolock[i][1], bigbolock[i][2]])

    return pix

start_time = time.time()
img = read_image("E:/Github/image-encryption-master/chaotic_map_encryption/chengshi.jpg")

img = np.resize(img, (864, 864, 3))
h, w = img.shape[0], img.shape[1]
enc_image = copy.deepcopy(img)
enc_image_HSV = copy.deepcopy(img)
enc_image_HSV_pixe = copy.deepcopy(img)
end_img = copy.deepcopy(img)
Big_num_blocks = img.shape[0] * img.shape[1] // BLOCK_SIZE
keyb = generate_keyb(Big_num_blocks)
keys = generate_keys(4)
keyp = generate_keyp(9)
keyv = generate_keyv(9)

for i in range(Big_num_blocks):
    x = keyb[i]
    start = i * BLOCK_SIZE
    start_HSV = i * BLOCK_SIZE
    row = start // w
    row_HSV = start_HSV // w
    #        col_start = (i % 2) * BLOCK_SIZE
    col_start = i * BLOCK_SIZE
    col_start_HSV = i * BLOCK_SIZE
    if col_start_HSV >= w:
        col_start_HSV = ((col_start_HSV % w) // block_size) * 9
    if col_start >= w:
        col_start = ((col_start % w) // BLOCK_SIZE) * 36
    if col_start != w:
        Bigblock = get_image_big_block(img, x)
        for image_bit in Bigblock:
            enc_image[row][col_start][0] = image_bit[0]
            enc_image[row][col_start][1] = image_bit[1]
            enc_image[row][col_start][2] = image_bit[2]
            col_start += 1
        hsv_image = cv2.cvtColor(enc_image, cv2.COLOR_BGR2HSV)
        Bigblock_HSV = get_image_big_block(hsv_image, i)
        for y in range(4):
            Smallblock = get_image_small_block(Bigblock_HSV, keys[y])
            for image_bit in Smallblock:
                enc_image_HSV[row_HSV][col_start_HSV][0] = image_bit[0]
                enc_image_HSV[row_HSV][col_start_HSV][1] = image_bit[1]
                enc_image_HSV[row_HSV][col_start_HSV][2] = image_bit[2]
                col_start_HSV += 1
        pixe_HSV_big = get_image_big_block(enc_image, i)
        col_start_HSV = col_start_HSV - 36
        for z in range(4):
            pixe_HSV_small = get_image_small_block(pixe_HSV_big, z)
            for k in range(9):
                img_bit_pixe = pixe_HSV_small[keyp[k]]
                enc_image_HSV_pixe[row_HSV][col_start_HSV][0] = img_bit_pixe[0]
                enc_image_HSV_pixe[row_HSV][col_start_HSV][1] = img_bit_pixe[1]
                enc_image_HSV_pixe[row_HSV][col_start_HSV][2] = img_bit_pixe[2]
                end_img[row_HSV][col_start_HSV][0] = ope[keyv[k]].encrypt(int(enc_image_HSV_pixe[row_HSV][col_start_HSV][0]))
                end_img[row_HSV][col_start_HSV][1] = ope[keyv[k]].encrypt(int(enc_image_HSV_pixe[row_HSV][col_start_HSV][1]))
                end_img[row_HSV][col_start_HSV][2] = ope[keyv[k]].encrypt(int(enc_image_HSV_pixe[row_HSV][col_start_HSV][2]))
                col_start_HSV += 1



end_time = time.time()
execution_time = end_time - start_time
print(f"程序运行时间为: {execution_time} 秒")
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\jiami.jpeg", enc_image)
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\jiami_HSV.jpeg", enc_image_HSV)
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\jiami_HSV_pixe.jpeg", enc_image_HSV_pixe)
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\final.jpeg", end_img)