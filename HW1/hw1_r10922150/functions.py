import cv2
import numpy as np
import matplotlib.pyplot as plt


def global_hist_equalization(image):  # 只能傳灰階
    row, col = image.shape
    result = np.zeros((row, col), np.uint8)
    # 先累計出cdf函數
    # 用一個255長度的list存在cdf裡面 np.uint8 灰階的顏色深度
    cdf = np.zeros(256, dtype=int)

    for each_row in image:
        for each_element in each_row:
            cdf[each_element] += 1

    cdf_sum = 0
    for cnt in range(len(cdf)):
        cdf[cnt] += cdf_sum
        cdf_sum = cdf[cnt]
    cdf_min, cdf_max = cdf[0], cdf[255]
    # 根據 cdf 存的function 跟cdf_min, cdf_max 參考網路上公式計算校正後的值
    for rownum in range(row):
        for colnum in range(col):
            result[rownum, colnum] = round(((cdf[image[rownum, colnum]] - cdf_min) / (
                cdf_max - cdf_min)) * (len(cdf) - 1))  # len(cdf) 是灰階的色彩深度 np.uint8
    return result


def local_hist_equalization(image, window_size):
    row, col = image.shape
    result = np.zeros((row, col), np.uint8)
    mirror_image = cv2.copyMakeBorder(
        image, window_size, window_size, window_size, window_size, cv2.BORDER_REFLECT)
    # 針對每個陣列作cdf
    for rownum in range(row):
        for colnum in range(col):
            # 算cdf
            base = image[rownum, colnum]
            rank = 0
            number_of_pixel = pow(window_size*2+1, 2)
            for row_window in range(-window_size, window_size+1):
                for col_window in range(-window_size, window_size+1):
                    if mirror_image[row_window+rownum, col_window+colnum] > base:
                        rank += 1
            result[rownum, colnum] = rank * 255 / number_of_pixel

    return result


def PSNR(image1,image2):
    # reference https://zhuanlan.zhihu.com/p/50757421
    diff = image1 - image2
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(255 * 255 / mse)
    return psnr

def plot_hist(file_name, image1, image2, image3, image1_title, image2_title, image3_title):


    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.title.set_text(image1_title)
    ax1.hist(image1.ravel(), bins=256)
    ax1.set_xlim([0, 255])

    ax2.title.set_text(image2_title)
    ax2.hist(image2.ravel(), bins=256)
    ax2.set_xlim([0, 255])

    ax3.title.set_text(image3_title)
    ax3.hist(image3.ravel(), bins=256)
    ax3.set_xlim([0, 255])

    plt.savefig(file_name)

def median_filter():

    pass