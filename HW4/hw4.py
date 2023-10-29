import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import *


def problem_1_a(image):
    I2 = np.array([[1, 2], [3, 0]], dtype='uint8')
    N = len(I2)
    # lecture06 p.15 threhold matrix
    threshold = 255 * (I2 + 0.5) / (N * N)

    h, w = image.shape
    threshold_matrix = np.tile(threshold, (h//N, w//N))
    result1 = np.zeros((h, w))
    result1[image >= threshold_matrix] = 1

    # result_1 = np.zeros((h, w))
    # for i in range(h):
    #     for j in range(w):
    #         if image[i, j] >= threshold_matrix[i&255, j&255]:
    #             result_1[i, j] = 1

    cv2.imwrite("result1.png", result1*255)


def problem_1_b(image):
    n = 2
    I = np.array([[1, 2], [3, 0]], dtype='uint8')

    for i in range(7):
        I2 = np.zeros((n*2, n*2))

        I2[0:n, 0:n] = I*4 + 1
        I2[0:n, n:] = I*4 + 2
        I2[n:, 0:n] = I*4 + 3
        I2[n:, n:] = I*4 + 0

        I = I2
        n *= 2

    I256 = I
    N = len(I256)
    threshold_matrix = 255 * (I256 + 0.5) / (N * N)

    h, w = image.shape
    result2 = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if image[i, j] >= threshold_matrix[i & 255, j & 255]:
                result2[i, j] = 1
    # cv2.imshow("result2", result2)
    cv2.imwrite("result2.png", result2*255)


def problem_1_c(image):
    # Floyd Steinberg
    result3 = np.copy(np.lib.pad(image, (1, 1), 'constant')) / 255

    kernel = [[0, 0, 7/16],
              [3/16, 5/16, 1/16]]

    ones = np.ones((2, 3))

    h, w = result3.shape

    for y in range(1, h-1):
        for x in range(1, w-1):

            old_value = result3[y, x]
            new_value = 0
            if (old_value >= 0.5):
                new_value = 1

            Error = old_value - new_value

            patch = result3[y:y+2, x-1:x+2]

            NewNumber = patch + Error * ones * kernel
            NewNumber[NewNumber > 1] = 1
            NewNumber[NewNumber < 0] = 0

            result3[y:y+2, x-1:x+2] = NewNumber
            result3[y, x] = new_value

    result3 = result3[1:623, 1:623]

    # Jarvis

    result4 = np.copy(np.lib.pad(image, (2, 2), 'constant')) / 255

    kernel = np.array([[0, 0, 0, 7, 5],
                       [3, 5, 7, 5, 3],
                       [1, 3, 5, 3, 1]])/48

    ones = np.ones((3, 5))

    h, w = result4.shape

    for y in range(2, h-2):
        for x in range(2, w-2):

            old_value = result4[y, x]
            new_value = 0
            if (old_value >= 0.5):
                new_value = 1

            Error = old_value - new_value

            patch = result4[y:y+3, x-2:x+3]

            NewNumber = patch + Error * ones * kernel
            NewNumber[NewNumber > 1] = 1
            NewNumber[NewNumber < 0] = 0

            result4[y:y+3, x-2:x+3] = NewNumber
            result4[y, x] = new_value

    result4 = result4[2:624, 2:624]

    cv2.imwrite("result3.png", result3*255)
    cv2.imwrite("result4.png", result4*255)
    # cv2.imshow("result3", result3)
    # cv2.imshow("result4", result4)
    # cv2.waitKey()
    # cv2.destroyAllWindows()


def problem_2_a(image):
    result2_1_image = cv2.resize(image, dsize=[0, 0], fx=0.3, fy=0.3)
    result2_2_image = cv2.resize(image, dsize=[0, 0], fx=0.5, fy=0.5)
    result2_3_image = cv2.resize(image, dsize=[0, 0], fx=0.8, fy=0.8)

    print(result2_1_image.shape)
    print(result2_2_image.shape)
    print(result2_3_image.shape)
    print(image.shape)
    cv2.imshow("result2_1", result2_1_image)
    cv2.imshow("result2_2", result2_2_image)
    cv2.imshow("result2_3", result2_3_image)
    cv2.imshow("original", image)
    cv2.imwrite("result2_1.png", result2_1_image)
    cv2.imwrite("result2_2.png", result2_2_image)
    cv2.imwrite("result2_3.png", result2_3_image)
    cv2.imwrite("original.png", image)

    cv2.waitKey()
    cv2.destroyAllWindows()


def problem_2_b(image):

    fourier_spectrum = DFT(image)
    output = np.log(1 + np.abs(fourier_spectrum))
    high_pass_result = HP(fourier_spectrum, 25, 2)
    output4 = np.abs(high_pass_result)
    cv2.imwrite("result7.png", output*10)
    # print(output*10)
    cv2.imwrite("result8.png", output4)
    cv2.imwrite("test.png", 10*np.log(1+np.abs(high_pass_result)))


def problem_2_c(image):
    fourier_spectrum = DFT(image)
    print(fourier_spectrum.shape)
    print(fourier_spectrum[0,0])
    output = np.log(1 + np.abs(fourier_spectrum))

    low_pass_result25 = LP(fourier_spectrum, 50, 2)
    low_pass_result35 = LP(fourier_spectrum, 60, 2)
    low_pass_result50 = LP(fourier_spectrum, 70, 2)
    low_pass_result80 = LP(fourier_spectrum, 80, 2)

    output2 = np.abs(low_pass_result25)
    output3 = np.abs(low_pass_result35)
    output4 = np.abs(low_pass_result50)
    output5 = np.abs(low_pass_result80)
    cv2.imwrite("2c_fft_spectrum.png", output*10)  
    cv2.imwrite("2c_low_pass_result.png", output2)
    cv2.imwrite("2c_low_pass_result2.png", output3)
    cv2.imwrite("2c_low_pass_result3.png", output4)
    cv2.imwrite("2c_low_pass_result4.png", output5)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    sample1 = cv2.imread("./hw4_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    sample2 = cv2.imread("./hw4_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    sample3 = cv2.imread("./hw4_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
    # problem_1_a(sample1)
    # problem_1_b(sample1)
    # problem_1_c(sample1)
    # problem_2_a(sample2)
    # problem_2_b(sample2)
    problem_2_c(sample3)
