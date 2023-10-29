import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import *


def problem_1_a(image):
    mask = [[255, 255, 255], [255, 255, 255], [255, 255, 255]]
    hit_image = on_mask_hit(image, mask)
    result = image - hit_image
    cv2.imwrite('result1.png', result)


def problem_1_b(image):
    result_image = image.copy()
    fill_mask = [[-1, 0],
                 [0, 1], [0, 0], [0, -1],
                 [1, 0]]
    position_list = [[0, 0]]
    cnt = 0
    result_image[0][0] = 255
    while(cnt <len(position_list)):
        ir, ic = position_list[cnt]
        # print(ir,ic)
        flag = True
        for each_mask in fill_mask:
            mr,mc = each_mask
            # print(mr,mc)
            if 0<=ir+mr<image.shape[0] and 0<=ic+mc<image.shape[1] and result_image[ir+mr][ic+mc] == 0:
                position_list.append([ir+mr,ic+mc])
                result_image[ir+mr][ic+mc] = 255
        cnt+=1
    # 此時得到的result image是把外面補起來 所以在把影像反轉過來
    for i in range(result_image.shape[0]):
        for j in range (result_image.shape[1]):
            if result_image[i][j]==0:
                result_image[i][j] = 255
            else:
                result_image[i][j] = 0

    cv2.imwrite('result2.png',image + result_image)




def problem_1_d(image):
    kernel_mask = [[-1, 0],
                   [0, 1], [0, 0], [0, -1],
                   [1, 0]]
    open_image = opening(image, kernel_mask)
    close_image = closing(image, kernel_mask)
    cv2.imwrite('result3.png', open_image)
    cv2.imwrite('result4.png', close_image)


def problem_2_a(image):
    # Law's method
    window_size = 13  # p23的參數
    F = image.copy().astype(np.float64)

    # p15的參數
    H1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 36
    H2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 12
    H3 = np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 12
    H4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 12
    H5 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    H6 = np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]]) / 4
    H7 = np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12
    H8 = np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 4
    H9 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4
    # step 1: Convolution p15的步驟
    M1 = convolution(F, H1)
    M2 = convolution(F, H2)
    M3 = convolution(F, H3)
    M4 = convolution(F, H4)
    M5 = convolution(F, H5)
    M6 = convolution(F, H6)
    M7 = convolution(F, H7)
    M8 = convolution(F, H8)
    M9 = convolution(F, H9)
    # step 2: Energy computation p21的步驟
    S = np.ones((window_size, window_size))
    T1 = convolution(M1 * M1, S)
    T2 = convolution(M2 * M2, S)
    T3 = convolution(M3 * M3, S)
    T4 = convolution(M4 * M4, S)
    T5 = convolution(M5 * M5, S)
    T6 = convolution(M6 * M6, S)
    T7 = convolution(M7 * M7, S)
    T8 = convolution(M8 * M8, S)
    T9 = convolution(M9 * M9, S)
    local_feature = np.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])  # (9, 600, 900)
    cnt = 1
    for T in local_feature:
        plt.imshow(T, cmap='gray')
        # plt.show()
        plt.imsave(str(cnt)+'.png', T, cmap='gray')
        cnt += 1
    return


def problem_2_b(image):
    # Law's method
    window_size = 13  # p23的參數
    F = image.copy().astype(np.float64)

    # p15的參數
    H1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 36
    H2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 12
    H3 = np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 12
    H4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 12
    H5 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    H6 = np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]]) / 4
    H7 = np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12
    H8 = np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 4
    H9 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 4
    # step 1: Convolution p15的步驟
    M1 = convolution(F, H1)
    M2 = convolution(F, H2)
    M3 = convolution(F, H3)
    M4 = convolution(F, H4)
    M5 = convolution(F, H5)
    M6 = convolution(F, H6)
    M7 = convolution(F, H7)
    M8 = convolution(F, H8)
    M9 = convolution(F, H9)
    # step 2: Energy computation p21的步驟
    S = np.ones((window_size, window_size))
    T1 = convolution(M1 * M1, S)
    T2 = convolution(M2 * M2, S)
    T3 = convolution(M3 * M3, S)
    T4 = convolution(M4 * M4, S)
    T5 = convolution(M5 * M5, S)
    T6 = convolution(M6 * M6, S)
    T7 = convolution(M7 * M7, S)
    T8 = convolution(M8 * M8, S)
    T9 = convolution(M9 * M9, S)
    local_feature = np.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])  # (9, 600, 900)
    law_method = np.moveaxis(local_feature, 0, -1)  # (600, 900, 9)

    cluster = kMeans(law_method.reshape(-1, 9), 4, 10)
    cluster = cluster.reshape(law_method.shape[0], law_method.shape[1])
    result5 = np.zeros((cluster.shape[0], cluster.shape[1], 3))
    # print(result5)
    # output color image
    for cluster_index in np.unique(cluster):
        position = np.argwhere(cluster == cluster_index)
        result5[position[:, 0], position[:, 1], :] = np.array((plt.cm.Set1(cluster_index)[0], plt.cm.Set1(cluster_index)[1], plt.cm.Set1(cluster_index)[2])) * 255
    result5 = result5.astype(np.uint8)
    cv2.imwrite('result5.png', result5)


def problem_2_c(image):
    # Law's method
    window_size = 13  # p23的參數
    F = image.copy().astype(np.float64)

    # p15的參數 調整比例
    H1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 12
    H2 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 36
    H3 = np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 4
    H4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 36
    H5 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4
    H6 = np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]]) / 12
    H7 = np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 12
    H8 = np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) / 36
    H9 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 36
    # step 1: Convolution p15的步驟
    M1 = convolution(F, H1)
    M2 = convolution(F, H2)
    M3 = convolution(F, H3)
    M4 = convolution(F, H4)
    M5 = convolution(F, H5)
    M6 = convolution(F, H6)
    M7 = convolution(F, H7)
    M8 = convolution(F, H8)
    M9 = convolution(F, H9)
    # step 2: Energy computation p21的步驟
    S = np.ones((window_size, window_size))
    T1 = convolution(M1 * M1, S)
    T2 = convolution(M2 * M2, S)
    T3 = convolution(M3 * M3, S)
    T4 = convolution(M4 * M4, S)
    T5 = convolution(M5 * M5, S)
    T6 = convolution(M6 * M6, S)
    T7 = convolution(M7 * M7, S)
    T8 = convolution(M8 * M8, S)
    T9 = convolution(M9 * M9, S)

    local_feature2 = np.stack([T1, T2, T3, T4, T5, T6, T7, T8, T9])  # (9, 600, 900)
    law_method2 = np.moveaxis(local_feature2, 0, -1)  # (600, 900, 9)

    cluster = kMeans(law_method2.reshape(-1, 9), 4, 10)
    cluster = cluster.reshape(law_method2.shape[0], law_method2.shape[1])
    result6 = np.zeros((cluster.shape[0], cluster.shape[1], 3))

    # output color image
    for cluster_index in np.unique(cluster):
        position = np.argwhere(cluster == cluster_index)
        result6[position[:, 0], position[:, 1], :] = np.array((plt.cm.Set1(cluster_index)[0], plt.cm.Set1(cluster_index)[1], plt.cm.Set1(cluster_index)[2])) * 255
    result6 = result6.astype(np.uint8)
    cv2.imwrite('result6.png', result6)


if __name__ == '__main__':

    sample1 = cv2.imread("./hw3_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    sample2 = cv2.imread("./hw3_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    sample3 = cv2.imread("./hw3_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
    problem_1_a(sample1)
    problem_1_b(sample1)

    problem_1_d(sample1)

    problem_2_a(sample2)
    problem_2_b(sample2)
    problem_2_c(sample2)
