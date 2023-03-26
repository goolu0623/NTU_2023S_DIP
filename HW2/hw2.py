import cv2
import numpy as np
import matplotlib.pyplot as plt
from functions import *


def problem_1_a():
    sample1 = cv2.imread("hw2_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    row_mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    col_mask = np.rot90(row_mask)
    threshold = 38

    result1 = sample1.copy()
    result2 = sample1.copy()
    theta_image = sample1.copy().astype(np.float)
    imageBoarded = Padding('DUPLICATE', 1, 1, 1, 1, result1)
    for i in range(len(result1)):
        for j in range(len(result1[i])):
            row_sum, col_sum = 0, 0
            for r in range(len(row_mask)):
                for c in range(len(row_mask[r])):
                    row_sum += imageBoarded[i + r - 1][j + c - 1] * row_mask[r][c]
                    col_sum += imageBoarded[i + r - 1][j + c - 1] * col_mask[r][c]
            theta_pixel = math.atan(col_sum / row_sum)
            gradientMagnitude = math.sqrt(row_sum ** 2 + col_sum ** 2)
            result1[i][j] = gradientMagnitude
            result2[i][j] = 0 if gradientMagnitude >= threshold else 255
            theta_image[i][j] =  theta_pixel

    cv2.imshow('1', result1)
    cv2.imshow('2', result2)
    cv2.waitKey()
    cv2.destroyAllwindows()
    return


def problem_1_b():
    sample1 = cv2.imread("hw2_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    '''
    canny edge detection: 
    five steps
    1. Noise reduction
    2. Compute gradient magnitude and orientation
    3. Non-maximal suppression
    4. Hysteretic thresholding
    5. Connected component labeling method
    '''
    gaus_image = GaussianFilter(sample1, K_size=5, sigma=1.3)
    grad_image =
    pass


def problem_1_c():
    pass


def problem_1_d():
    pass


def problem_1_e():
    pass


def problem_2_a():
    pass


def problem_2_b():
    pass


def test():
    sample1 = cv2.imread("hw2_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    cv2.imshow('test', GaussianFilter(sample1))
    cv2.imshow('samp', sample1)
    cv2.waitKey()
    cv2.destroyAllwindows()


if __name__ == '__main__':
    test()
    # problem_1_a()
