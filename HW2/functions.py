import cv2
import numpy as np
import math


# def Sobel(image, mask_size, k_value):
#     row, col = image.shape
#     for row_num in range(row):
#         for col_num in range(col):
#     pass


# def Prewitt(image, threshold, p1mask, p2mask):
#     PrewittImage = image.copy()
#     imageBoarded = Padding('DUPLICATE', 1, 1, 1, 1, PrewittImage)
#     for i in range(len(image)):
#         for j in range(len(image[i])):
#             p1sum, p2sum = 0, 0
#             for r in range(len(p1mask)):
#                 for c in range(len(p1mask[r])):
#                     p1sum += imageBoarded[i + r - 1][j + c - 1] * p1mask[r][c]
#                     p2sum += imageBoarded[i + r - 1][j + c - 1] * p2mask[r][c]
#             gradientMagnitude = math.sqrt(p1sum ** 2 + p2sum ** 2)
#             PrewittImage[i][j] = 0 if gradientMagnitude >= threshold else 255
#     return PrewittImage

def GaussianFilter(img,K_size, sigma):
    h, w = img.shape



    # duplicate padding
    pad = K_size // 2
    out = Padding('DUPLICATE', pad,pad,pad,pad, img).astype(np.float)

    # out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float)
    # out[pad:pad + h, pad:pad + w] = img.copy().astype(np.float)

    # 定义滤波核
    K = np.zeros((K_size, K_size), dtype=np.float)

    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (sigma * np.sqrt(2 * np.pi))
    K /= K.sum()

    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            out[pad + y, pad + x] = np.sum(K * tmp[y:y + K_size, x:x + K_size])

    out = out[pad:pad + h, pad:pad + w].astype(np.uint8)

    return out


def Padding(padding_type, left, right, top, bottom, image):
    # 先只能吃灰階

    # 預計實作: zero, duplicate, mirror
    row, col = image.shape[0], image.shape[1]
    Result_image = np.zeros((row + top + bottom, col + left + right), np.uint8)
    # ZERO padding:
    if padding_type == 'ZERO':
        for row_num in range(row):
            for col_num in range(col):
                Result_image[row_num + top][col_num + left] = image[row_num][col_num]

    # DUPLICATE padding:
    if padding_type == 'DUPLICATE':
        for row_num in range(row):
            for col_num in range(col):
                Result_image[row_num + top][col_num + left] = image[row_num][col_num]
        # 左
        for row_num in range(row):
            for col_num in range(left):
                Result_image[row_num + top][col_num] = image[row_num][0]
        # 右
        for row_num in range(row):
            for col_num in range(right):
                Result_image[row_num + top][col_num + col + left] = image[row_num][col - 1]
        # 上
        for row_num in range(top):
            for col_num in range(col):
                Result_image[row_num][col_num + left] = image[0][col_num]
        # 下
        for row_num in range(bottom):
            for col_num in range(col):
                Result_image[row + row_num + top][col_num + left] = image[row - 1][col_num]

    # Mirror padding:
    # if padding_type == 'MIRROR':
    #     for row_num in range(row):
    #         for col_num in range(col):
    #             Result_image[row_num+top][col_num+left] = image[row_num][col_num]
    #
    #     # top
    #     for row_num in range(top):
    #         for col_num in range(col):
    #             Result_image[-row_num+top][col_num+left] = image[-row_num+top]
    #     # bottom
    #     # left
    #     # right
    #     # 四個角落
    return Result_image
