import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import *


def problem_0_a():
    '''
    reference: http://atlaboratary.blogspot.com/2013/08/rgb-g-rey-l-gray-r0.html
    對於彩色轉灰度，有一個很著名的心理學公式：
    float算法 : Gray = R*0.299 + G*0.587 + B*0.114
    int  算法 : Gray = (R*299 + G*587 + B*114 + 500) / 1000
    '''
    sample1 = cv2.imread("hw1_sample_images/sample1.png")

    row, column, channels = sample1.shape
    result1 = np.zeros((row, column, 1), np.uint8)
    # for each in sample1:
    #     print(each)
    for rownum in range(row):
        for colnum in range(column):
            R, G, B = sample1[rownum, colnum]
            result1[rownum, colnum] = (
                R * 299 + G * 587 + B * 114 + 500) / 1000
    cv2.imwrite("result1.png", result1)


def problem_0_b():
    result1 = cv2.imread("result1.png")
    row, column, channel = result1.shape
    result2 = np.zeros((row, column, 3), np.uint8)
    for rownum in range(row):
        for colnum in range(column):
            result2[rownum, colnum] = result1[row - 1 - rownum, colnum]
    cv2.imwrite("result2.png", result2)


def problem_1_a():
    result2 = cv2.imread("result2.png")
    row, column, channel = result2.shape
    result3 = np.zeros((row, column, 3), np.uint8)
    for rownum in range(row):
        for colnum in range(column):
            result3[rownum, colnum] = result2[rownum, colnum] / 3
    cv2.imwrite("result3.png", result3)


def problem_1_b():
    result3 = cv2.imread("result3.png")
    row, column, channel = result3.shape
    result4 = np.zeros((row, column, 3), np.uint8)
    for rownum in range(row):
        for colnum in range(column):
            result4[rownum, colnum] = result3[rownum, colnum] * 3
    cv2.imwrite("result4.png", result4)


def problem_1_c():
    sample2 = cv2.imread("./hw1_sample_images/sample2.png",
                         cv2.IMREAD_GRAYSCALE)
    result3 = cv2.imread("result3.png", cv2.IMREAD_GRAYSCALE)
    result4 = cv2.imread("result4.png", cv2.IMREAD_GRAYSCALE)

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.title.set_text("sample2 hist")
    ax1.hist(sample2.ravel(), bins=256)
    ax1.set_xlim([0, 255])

    ax2.title.set_text("result3 hist")
    ax2.hist(result3.ravel(), bins=256)
    ax2.set_xlim([0, 255])

    ax3.title.set_text("result4 hist")
    ax3.hist(result4.ravel(), bins=256)
    ax3.set_xlim([0, 255])

    plt.savefig("hist_compare.png")


def problem_1_d():
    sample2 = cv2.imread("./hw1_sample_images/sample2.png",
                         cv2.IMREAD_GRAYSCALE)
    result3 = cv2.imread("result3.png", cv2.IMREAD_GRAYSCALE)
    result4 = cv2.imread("result4.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("result5.png", global_hist_equalization(sample2))
    cv2.imwrite("result6.png", global_hist_equalization(result3))
    cv2.imwrite("result7.png", global_hist_equalization(result4))


def problem_1_e():
    sample2 = cv2.imread("./hw1_sample_images/sample2.png",
                         cv2.IMREAD_GRAYSCALE)
    result3 = cv2.imread("result3.png", cv2.IMREAD_GRAYSCALE)
    result4 = cv2.imread("result4.png", cv2.IMREAD_GRAYSCALE)
    cv2.imwrite("result8.png",
                local_hist_equalization(sample2, 50))
    cv2.imwrite("result9.png",
                local_hist_equalization(result3, 50))
    cv2.imwrite("result10.png",
                local_hist_equalization(result4, 50))


def problem_1_f():
    # 嘗試過local HE的各種參數如report敘述，這邊因為結果不佳，為了加快速度因此暫時註解掉
    pass

def problem_2_a():

    sample3 = cv2.imread("./hw1_sample_images/sample3.png",
                         cv2.IMREAD_GRAYSCALE)
    sample4 = cv2.imread("./hw1_sample_images/sample4.png",
                         cv2.IMREAD_GRAYSCALE)
    sample5 = cv2.imread("./hw1_sample_images/sample5.png",
                         cv2.IMREAD_GRAYSCALE)

    # low pass filter on sample 4
    row, col = sample4.shape
    # 因為kernel用3*3 所以mirror image上下左右都+1
    mirror_image = cv2.copyMakeBorder(
        sample4, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result12 = np.zeros((row, col))
    for rownum in range(row):
        for colnum in range(col):
            sum = 0
            kernel_mask = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
            for rowker in range(-1, 2):
                for colker in range(-1, 2):
                    sum += mirror_image[rownum+rowker, colnum +
                                        colker] * kernel_mask[rowker+1][colker+1]
            result12[rownum, colnum] = sum/16
    cv2.imwrite("result12.png", result12)
    # median filter on sample 5
    row, col = sample5. shape
    mirror_image = cv2.copyMakeBorder(
        sample5, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    result13 = np.zeros((row, col))
    for rownum in range(row):
        for colnum in range(col):
            median_list = []
            for rowker in range(-1, 2):
                for colker in range(-1, 2):
                    median_list.append(
                        mirror_image[rownum+rowker, colnum+colker])
            median_list.sort()
            result13[rownum, colnum] = median_list[4]
    cv2.imwrite("result13.png", result13)


def problem_2_b():

    sample3 = cv2.imread("./hw1_sample_images/sample3.png",
                         cv2.IMREAD_GRAYSCALE)
    result12 = cv2.imread("result12.png", cv2.IMREAD_GRAYSCALE)
    result13 = cv2.imread("result13.png", cv2.IMREAD_GRAYSCALE)
    
    sample4 = cv2.imread("./hw1_sample_images/sample4.png",
                         cv2.IMREAD_GRAYSCALE)
    sample5 = cv2.imread("./hw1_sample_images/sample5.png",
                         cv2.IMREAD_GRAYSCALE)
    print(PSNR(sample3, result12))
    print(PSNR(sample3, result13))
    print(PSNR(sample3, sample4))
    print(PSNR(sample3, sample5))



def test():
    image1 = cv2.imread("result8.png", cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread("result9.png", cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread("result10.png", cv2.IMREAD_GRAYSCALE)

    fig = plt.figure()
    fig.set_figheight(5)
    fig.set_figwidth(15)
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.title.set_text("result8 hist")
    ax1.hist(image1.ravel(), bins=256)
    ax1.set_xlim([0, 255])

    ax2.title.set_text("result9 hist")
    ax2.hist(image2.ravel(), bins=256)
    ax2.set_xlim([0, 255])

    ax3.title.set_text("result10 hist")
    ax3.hist(image3.ravel(), bins=256)
    ax3.set_xlim([0, 255])

    plt.savefig("result8910_compare.png")


if __name__ == '__main__':

    problem_0_a()
    problem_0_b()
    # problem_1_a()
    # problem_1_b()
    # problem_1_c()
    # problem_1_d()
    # problem_1_e()
    # problem_1_f()
    # problem_2_a()
    # problem_2_b()
    # test()
