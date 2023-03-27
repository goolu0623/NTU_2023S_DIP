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
    2. Compute gradient magnitude and orientation
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
    gauss_image = GaussianFilter(image, 3, 0.01)

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
    gauss_image = GaussianFilter(image, K_size=3, sigma=0.1)

    # step2: Compute Gradient Magnitude and Orientation
    grad_image = gradient_image(image=gauss_image, type='SOBEL')
    ori_image = orientation_image(image=gauss_image, type='SOBEL')

    # step3: Non-maximal suppression
    non_max_image = non_maximal_suppression(grad_image, ori_image)

    # step4: Hysteretic thresholding
    threshold_image = hysteretic_thresholding(non_max_image, 150, 50)

    # step5: Connected Component Labeling Method
    canny_image = connected_component_labeling(threshold_image)

    cv2.imwrite('result6.png', canny_image)

    cv2.imwrite('result7.png', hough_transform(canny_image, 100))


def problem_2_a():
    pass


def problem_2_b():
    pass


def test():

    pass


if __name__ == '__main__':

    sample1 = cv2.imread("./hw2_sample_images/sample1.png", cv2.IMREAD_GRAYSCALE)
    sample2 = cv2.imread("./hw2_sample_images/sample2.png", cv2.IMREAD_GRAYSCALE)
    sample3 = cv2.imread("./hw2_sample_images/sample3.png", cv2.IMREAD_GRAYSCALE)
    sample5 = cv2.imread("./hw2_sample_images/sample5.png", cv2.IMREAD_GRAYSCALE)
    # # test()
    # problem_1_a(sample1)
    # problem_1_b(sample1)
    # problem_1_c(sample1)
    # problem_1_d(sample2)
    # result5 = cv2.imread("./result5.png", cv2.IMREAD_GRAYSCALE)
    # problem_1_e(result5)
    # result6 = cv2.imread("./result6.png", cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite('hough1.png', hough_transform(result6, 1))
    cvguass = cv2.GaussianBlur(sample2, (3,3), 0)
    cvcanny = cv2.Canny(cvguass, 50, 150)
    lines = cv2.HoughLinesP(cvcanny, 1, np.pi/180, 1, np.array([]),10,1)
    line_image = np.copy(sample2)*0
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    combo = cv2.addWeighted(cvcanny, 0.8, line_image, 0.2, 1)
    cv2.imshow('hough', combo)
    cv2.imshow('canny', cvcanny)
    # cv2.imshow('result6',result6)
    # cv2.imshow('cv_hough', cvhough)
    cv2.waitKey()
    cv2.destroyAllWindows()