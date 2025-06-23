import numpy as np

BLOCK_SIZE = 16


def get_image_block(img, block_no):
    h, w = img.shape[0], img.shape[1]
    start = block_no * BLOCK_SIZE
    num_blocks = img.shape[0] * img.shape[1] // BLOCK_SIZE
    row = start // w
    #    col_start = (block_no%2)*BLOCK_SIZE
    col_start = block_no * BLOCK_SIZE
    if col_start > w:
        col_start = ((col_start % w) // BLOCK_SIZE)*16
    col_end = (col_start + BLOCK_SIZE)-1

    pix = []

    for i in range(col_start, col_end):
        pix.append([img[row][i][0], img[row][i][1], img[row][i][2]])

    return pix


a = np.zeros((48, 64, 3))

x = get_image_block(a, 50)
