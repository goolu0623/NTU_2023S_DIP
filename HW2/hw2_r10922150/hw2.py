import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import *


def problem_1_a(image):

    cv2.imwrite("result1.png", gradient_image(image, 'SOBEL'))
    cv2.imwrite("result2.png", threshld_image(gradient_image(image, 'SOBEL'), 100))


def problem_1_b(image):
    '''
    canny edge detection: 
    five steps
    1. Noise reduction
    2. Compute gradiji3m0v3ent magnitude and orientation
    3. Non-maximal suppression
    4. Hysteretic thresholding
    5. Connected component labeling method
    '''
    # step1: Noise Reduction
    gauss_image = GaussianFilter(image, K_size=3, sigma=0.01)

    # step2: Compute Gradient Magnitude and Orientation
    grad_image = gradient_image(image=gauss_image, type='SOBEL')
    ori_image = orientation_image(image=gauss_image, type='SOBEL')

    # step3: Non-maximal suppression
    non_max_image = non_maximal_suppression(grad_image, ori_image)

    # step4: Hysteretic thresholding
    threshold_image = hysteretic_thresholding(non_max_image, 100, 5)

    # step5: Connected Component Labeling Method
    canny_image = connected_component_labeling(threshold_image)

    cv2.imwrite('result3.png', canny_image)


def problem_1_c(image):

    # step1: Gaussian Filter
    gauss_image = GaussianFilter(image, 5, 1)

    # step2: 2nd-order Derivative(Laplacian)
    lp_image = Derivative_gradient(gauss_image)

    hist, bins = np.histogram(lp_image, bins=1000, range=(lp_image.min(), lp_image.max()))
    bins = bins[:-1]
    plt.bar(bins, hist)
    plt.title('hist')
    plt.savefig('hist.png')

    # step3: Thresholded
    row, col = lp_image.shape
    T = 1.2
    threshold_image = lp_image.copy()
    for i in range(len(threshold_image)):
        for j in range(len(threshold_image[i])):
            threshold_image[i][j] = 0 if lp_image[i][j] > T else 255

    # step4: Zero-crossing
    zero_image = np.zeros((row, col))
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if threshold_image[i][j] == 0:
                if (lp_image[i-1][j-1]*lp_image[i+1][j+1] < 0 or lp_image[i-1][j]*lp_image[i+1][j] < 0 or
                        lp_image[i-1][j+1]*lp_image[i+1][j-1] < 0 or lp_image[i][j-1]*lp_image[i][j+1] < 0):
                    zero_image[i][j] = 255

    cv2.imwrite('result4.png', zero_image)


def problem_1_d(image):
    # c值介於0.6~0.83之間
    image = image
    gauss_image = GaussianFilter(image, 10, 2)
    edge_crispening_image = edge_crispening(image, gauss_image, c=0.6)
    cv2.imwrite('result5.png', edge_crispening_image)


def problem_1_e(image):
    '''
    canny edge detection: 
    five steps
    1. Noise reduction
    2. Compute gradient magnitude and orientation
    3. Non-maximal suppression
    4. Hysteretic thresholding
    5. Connected component labeling method
    '''
    # step1: Noise Reduction
    gauss_image = GaussianFilter(image, K_size=5, sigma=1)

    # step2: Compute Gradient Magnitude and Orientation
    grad_image = gradient_image(image=gauss_image, type='SOBEL')
    ori_image = orientation_image(image=gauss_image, type='SOBEL')

    # step3: Non-maximal suppression
    non_max_image = non_maximal_suppression(grad_image, ori_image)

    # step4: Hysteretic thresholding
    threshold_image = hysteretic_thresholding(non_max_image, 70, 10)

    # step5: Connected Component Labeling Method
    canny_image = connected_component_labeling(threshold_image)

    cv2.imwrite('result6.png', canny_image)



def problem_2_a(image):

    col_his_array = np.zeros(len(image))
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 255:
                col_his_array[j] += 1
    col_start_num, col_end_num = 0, 0
    col_start_num2, col_end_num2 = 0, 0
    for i in range(len(col_his_array)-1):
        if col_his_array[i] == 0 and col_his_array[i+1] != 0:
            if col_start_num == 0:
                col_start_num = i+1
            else:
                col_start_num2 = i+1
        elif col_his_array[i] != 0 and col_his_array[i+1] == 0:
            if col_end_num == 0:
                col_end_num = i
            else:
                col_end_num2 = i
    # 現在找到狗的範圍是col_start_num到col_end_num
    # 再針對這個範圍內去找頭尾
    row_his_array = np.zeros(len(image[0]))
    for i in range(len(image)):
        for j in range(col_start_num2, col_end_num2+1):
            if image[i][j] < 255:
                row_his_array[i] += 1
    row_start_num, row_end_num = 0, 0
    for i in range(len(row_his_array)-1):
        if row_his_array[i] == 0 and row_his_array[i+1] != 0:
            row_start_num = i+1
        elif row_his_array[i] != 0 and row_his_array[i+1] == 0:
            row_end_num = i

    # 找餅乾row範圍
    row_his_array = np.zeros(len(image[0]))
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < 255:
                row_his_array[i] += 1
    row_start_num2, row_end_num2 = 0, 0
    for i in range(len(row_his_array)-1):
        if row_his_array[i] == 0 and row_his_array[i+1] != 0:
            row_start_num2 = i+1
        elif row_his_array[i] != 0 and row_his_array[i+1] == 0:
            row_end_num2 = i

    # 狗的範圍在col_start_num, col_end_num, row_start_num, row_end_num
    result = image.copy()
    cv2.rectangle(result, (col_start_num2, row_start_num), (col_end_num2, row_end_num), 0, 1)  # 原狗
    cv2.rectangle(result, (col_end_num, row_start_num-90), (col_end_num2+60, row_end_num2), 0, 1)  # 新狗
    # col_scaler = (col_end_num2+60-col_end_num) / (col_end_num2-col_start_num2)
    # row_scaler = (row_end_num2-row_start_num+90) / (row_end_num-row_start_num)
    # col_org = col_end_num2+60
    # row_org = row_start_num-90
    # col_target = col_end_num
    # row_target = row_end_num2
    # print(col_end_num2-col_start_num2, row_end_num-row_start_num)
    # print(col_end_num2+60-col_end_num, row_end_num2-row_start_num+90)
    # 要非線性變換 越靠右上型變越少
    output_image = image.copy()
    # 先算旋轉後座標
    # 再算線性變換位置

    # 原狗的中心

    center_x = (col_start_num2+col_end_num2) / 2
    center_y = (row_start_num + row_end_num) / 2
    for i in range(row_start_num, row_end_num):
        for j in range(col_start_num2, col_end_num2):
            output_image[i][j] = 255
    for i in range(row_start_num, row_end_num):
        for j in range(col_start_num2, col_end_num2):
            if image[i][j] < 255:
                output_x = (j-center_x) * np.cos(-np.pi/5) + (i-center_y)*np.sin(-np.pi/5)+center_x
                output_y = (i-center_y) * np.cos(-np.pi/5) - (j-center_y)*np.sin(-np.pi/5)+center_y
                output_x = 0.6*(output_x-center_x) + center_x
                output_y = 1.6*(output_y - center_y) + center_y
                output_image[int(output_y)][int(output_x)] = image[i][j]
    cv2.imwrite('rectangle.png', result)
    cv2.imwrite('result8.png', output_image)



def problem_2_b(image):
    # 中間的160*160往外擠壓一個圓型
    es = cv2.circle(image, (150, 170), 75, (255), 1)
    result = image.copy()
    # cv2.imshow('1', es)
    center_x, center_y = 150, 170

    # 放大力度
    R1 = int(np.sqrt(2*(75**2)))
    R1 = 80
    # print(R1)
    for i in range(-75, 76):
        for j in range(-75, 76):
            distance = np.sqrt(i**2+j**2)

            # 小於75才做型變
            # reference : https://blog.csdn.net/yangtrees/article/details/9095731
            if distance <= 75.0:
                new_x, new_y = int(i * np.power(distance/R1, 1.2) + center_x), int(j * np.power(distance/R1, 1.2) + center_y)
                result[j+center_y][i+center_x] = image[new_y][new_x]

    cv2.imwrite('result9.png', result)
    cv2.imwrite('circle.png',es)





if __name__ == '__main__':

    sample1 = cv2.imread("./hw2_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    sample2 = cv2.imread("./hw2_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    sample3 = cv2.imread("./hw2_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
    sample5 = cv2.imread("./hw2_sample_images/sample5.png", cv2.IMREAD_GRAYSCALE)
    problem_1_a(sample1)
    problem_1_b(sample1)
    problem_1_c(sample1)
    problem_1_d(sample2)
    result5 = cv2.imread("./result5.png", cv2.IMREAD_GRAYSCALE)
    problem_1_e(result5)

    problem_2_a(sample3)
    problem_2_b(sample5)

    # cv2.waitKey()
    # cv2.destroyAllWindows()
