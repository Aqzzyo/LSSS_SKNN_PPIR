import cupy as cp

import time

import cv2
import numpy as np
import torch
# import torchvision
from key_generate import generate_keyb
from key_generate import generate_keys
from key_generate import generate_keyp
from key_generate import generate_keyv


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
img = read_image("E:/Github/image-encryption-master/chaotic_map_encryption/lena.png")
BLOCK_SIZE = 36  # Example, adjust based on your use case

# Convert the image to a PyTorch tensor and move it to the GPU
img = torch.tensor(cv2.resize(img, (216, 216)), dtype=torch.float32).cuda()  # Move to GPU
h, w = img.shape[0], img.shape[1]
img_tensor = img.permute(2, 0, 1)
img_tensor = img_tensor.unsqueeze(0)

# Deep copies of the image on the GPU
enc_image = img.clone()
enc_image_HSV = img.clone()
enc_image_HSV_pixe = img.clone()
end_img = img.clone()

# Precomputing constants
Big_num_blocks = (img.shape[0] * img.shape[1]) // BLOCK_SIZE
keyb = generate_keyb(Big_num_blocks)  # Precomputed block keys
keys = generate_keys(4)  # Precomputed small block keys
keyp = generate_keyp(9)  # Precomputed pixel permutation keys
keyv = generate_keyv(9)  # Precomputed encryption keys

# GPU processing loop
for i in range(Big_num_blocks):
    x = keyb[i]
    start = i * BLOCK_SIZE
    start_HSV = i * BLOCK_SIZE
    row = start // w
    row_HSV = start_HSV // w

    col_start = i * BLOCK_SIZE
    col_start_HSV = i * BLOCK_SIZE
    ## 对图像进行分块处理，首先分成1296个6*6块，big_patch_img维度为【1，108，1296】batch_size，6*6*C，big_block_num,第二通道排列方式为36个1通道-36个二通道-36个三通道
    nn_Unfold = torch.nn.Unfold(kernel_size=(6, 6), dilation=1, padding=0, stride=6)
    big_patche_img = nn_Unfold(img_tensor)

    # Adjust column indices for blocks
    if col_start_HSV >= w:
        col_start_HSV = ((col_start_HSV % w) // BLOCK_SIZE) * 9
    if col_start >= w:
        col_start = ((col_start % w) // BLOCK_SIZE) * BLOCK_SIZE

    if col_start != w:
        # Fetch big block directly on GPU
        Bigblock = get_image_big_block(img, x)  # Ensure this function is GPU compatible

        # Process image big block
        for image_bit in Bigblock:
            enc_image[row, col_start, 0] = image_bit[0]
            enc_image[row, col_start, 1] = image_bit[1]
            enc_image[row, col_start, 2] = image_bit[2]
            col_start += 1

        # Convert to HSV (PyTorch doesn't support this directly, use CPU temporarily)
        hsv_image = cv2.cvtColor(enc_image.cpu().numpy().astype(np.uint8), cv2.COLOR_BGR2HSV)
        hsv_image = torch.tensor(hsv_image, dtype=torch.float32).cuda()  # Move back to GPU

        Bigblock_HSV = get_image_big_block(hsv_image, i)
        for y in range(4):
            Smallblock = get_image_small_block(Bigblock_HSV, keys[y])
            for image_bit in Smallblock:
                enc_image_HSV[row_HSV, col_start_HSV, 0] = image_bit[0]
                enc_image_HSV[row_HSV, col_start_HSV, 1] = image_bit[1]
                enc_image_HSV[row_HSV, col_start_HSV, 2] = image_bit[2]
                col_start_HSV += 1

        pixe_HSV_big = get_image_big_block(enc_image, i)
        col_start_HSV = col_start_HSV - BLOCK_SIZE
        for z in range(4):
            pixe_HSV_small = get_image_small_block(pixe_HSV_big, z)
            for k in range(9):
                img_bit_pixe = pixe_HSV_small[keyp[k]]
                enc_image_HSV_pixe[row_HSV, col_start_HSV, 0] = img_bit_pixe[0]
                enc_image_HSV_pixe[row_HSV, col_start_HSV, 1] = img_bit_pixe[1]
                enc_image_HSV_pixe[row_HSV, col_start_HSV, 2] = img_bit_pixe[2]

                # Encryption on GPU
                end_img[row_HSV, col_start_HSV, 0] = ope[keyv[k]].encrypt(
                    int(enc_image_HSV_pixe[row_HSV, col_start_HSV, 0].item()))
                end_img[row_HSV, col_start_HSV, 1] = ope[keyv[k]].encrypt(
                    int(enc_image_HSV_pixe[row_HSV, col_start_HSV, 1].item()))
                end_img[row_HSV, col_start_HSV, 2] = ope[keyv[k]].encrypt(
                    int(enc_image_HSV_pixe[row_HSV, col_start_HSV, 2].item()))
                col_start_HSV += 1

# Convert back to NumPy for saving or further processing
end_img = end_img.cpu().numpy().astype(np.uint8)
enc_image = enc_image.cpu().numpy().astype(np.uint8)
enc_image_HSV = enc_image_HSV.cpu().numpy().astype(np.uint8)
enc_image_HSV_pixe = enc_image_HSV_pixe.cpu().numpy().astype(np.uint8)
end_time = time.time()
execution_time = end_time - start_time
print(f"程序运行时间为: {execution_time} 秒")
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\jiami.jpeg", enc_image)
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\jiami_HSV.jpeg", enc_image_HSV)
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\jiami_HSV_pixe.jpeg", enc_image_HSV_pixe)
cv2.imwrite(r"E:\Github\PyRetri-master\HSV_img_encrption\enc_data\final.jpeg", end_img)