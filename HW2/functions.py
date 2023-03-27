import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def gradient_image(image, type):

    if type == 'SOBEL':
        K = 2
    elif type == 'PREWITT':
        K = 1
    col_mask = np.array([[1, K, 1], [0, 0, 0], [-1, -K, -1]])
    row_mask = np.rot90(col_mask)
    K_size = len(col_mask)
    pad = K_size//2
    grad_image = image.copy()
    pad_image = np.lib.pad(image,(pad,pad),'reflect')

    for i in range(len(grad_image)):
        for j in range(len(grad_image[i])):
            row_sum, col_sum = 0, 0
            for r in range(len(row_mask)):
                for c in range(len(row_mask[r])):
                    row_sum += pad_image[i + r ][j + c ] * row_mask[r][c]
                    col_sum += pad_image[i + r ][j + c ] * col_mask[r][c]
            gradientMagnitude = math.sqrt(row_sum ** 2 + col_sum ** 2)
            grad_image[i][j] = gradientMagnitude
    return grad_image


def threshld_image(image, threshold):
    result_image = image.copy()
    result_image[image > threshold] = 255
    result_image[image <= threshold] = 0
    return result_image


def orientation_image(image, type):
    if type == 'SOBEL':
        K = 2
    elif type == 'PREWITT':
        K = 1
    col_mask = np.array([[-1, -K, -1], [0, 0, 0], [1, K, 1]])
    row_mask = np.rot90(col_mask)
    K_size = len(col_mask)
    pad = K_size//2


    theta_image = image.copy().astype(float)
    pad_image = np.lib.pad(image,(pad,pad),'reflect')
    for i in range(len(image)):
        for j in range(len(image[i])):
            row_sum, col_sum = 0, 0
            for r in range(len(row_mask)):
                for c in range(len(row_mask[r])):
                    row_sum += pad_image[i + r - 1][j + c - 1] * row_mask[r][c]
                    col_sum += pad_image[i + r - 1][j + c - 1] * col_mask[r][c]
            theta_image[i][j] = math.atan(col_sum / row_sum) * 180.0 / math.pi
    return theta_image


def non_maximal_suppression(grad_image, ori_image):
    row = grad_image.shape[0]
    col = grad_image.shape[1]
    non_maximal = grad_image.copy()
    ori_image[ori_image < 0] += 180
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            G1 = 0
            G2 = 0
            # angle == 0
            if ((0 <= ori_image[i][j] < 22.5) or (157.5 < ori_image[i][j] <= 180)):
                G1 = grad_image[i][j + 1]
                G2 = grad_image[i][j - 1]
            # angle == 45
            elif (22.5 <= ori_image[i][j] <= 67.5):
                G1 = grad_image[i + 1][j - 1]
                G2 = grad_image[i - 1][j + 1]
            # angle == 90
            elif (67.5 < ori_image[i][j] < 112.5):
                G1 = grad_image[i + 1][j]
                G2 = grad_image[i - 1][j]
            # angle == 135
            elif (112.5 <= ori_image[i][j] <= 157.5):
                G1 = grad_image[i - 1][j - 1]
                G2 = grad_image[i + 1][j + 1]

            if (grad_image[i][j] >= G1) and (grad_image[i][j] >= G2):
                non_maximal[i][j] = grad_image[i][j]
            else:
                non_maximal[i][j] = 0
    return non_maximal


def hysteretic_thresholding(image, threshold_H, threshold_L):
    row, col = image.shape
    threshold_image = np.zeros((row, col), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            if (image[i][j] >= threshold_H):
                threshold_image[i][j] = 2
            elif (image[i][j] >= threshold_L):
                threshold_image[i][j] = 1
            else:
                threshold_image[i][j] = 0
    return threshold_image


def set_edge_point(image, i, j):  # recursively
    row, col = image.shape
    if i < 0 or j < 0 or i >= row or j >= col:
        return 0

    for k in range(3):
        for l in range(3):
            if (i + k - 1 >= 0 and i + k - 1 < row and j + l - 1 >= 0 and j + l - 1 < col):
                if (image[i + k - 1][j + l - 1] == 1):
                    image[i][j] = set_edge_point(image, i + k - 1, j + l - 1)
                if (image[i + k - 1][j + l - 1] == 2):
                    return 255
                else:
                    return 0


def connected_component_labeling(image):
    row, col = image.shape
    for i in range(row):
        for j in range(col):
            if (image[i][j] == 2):  # edge
                image[i][j] = 255
            elif (image[i][j] == 1):  # candidate
                set_edge_point(image, i, j)
    return image


def GaussianFilter(grad_image, K_size, sigma):
    # reference https://blog.csdn.net/qq_28368377/article/details/107288647
    # 從上面修改成我的padding方式+變成灰階版本
    h, w = grad_image.shape

    # duplicate padding
    pad = K_size // 2
    out = np.lib.pad(grad_image,(pad,pad),'reflect').astype(float)

    # out = np.zeros((h + 2 * pad, w + 2 * pad), dtype=np.float)
    # out[pad:pad + h, pad:pad + w] = grad_image.copy().astype(np.float)

    # kernel mask
    K = np.zeros((K_size, K_size), dtype=float)

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




def Derivative_gradient(image):

    lp_kernel = np.array(
        [[-2, 1, -2],
         [1, 4, 1],
         [-2, 1, -2]]
    )/8
    K_size = len(lp_kernel)
    pad = K_size//2

    pad_image = np.lib.pad(image,(pad,pad),'reflect')
    lp_image = np.zeros((image.shape), dtype=float)
    for i in range(len(image)):
        for j in range(len(image[i])):
            grad_sum = 0.0
            for r in range(len(lp_kernel)):
                for c in range(len(lp_kernel[r])):
                    grad_sum += pad_image[i + r - 1][j + c - 1] * lp_kernel[r][c]
            lp_image[i][j] = grad_sum
    return lp_image


def edge_crispening(org_image, gauss_image, c):
    edge_crispening_image = org_image.copy().astype(np.uint8)
    for i in range(len(org_image)):
        for j in range(len(org_image[i])):
            value = (c / (2.0 * c - 1)) * org_image[i][j] - ((1 - c) / (2.0 * c - 1)) * gauss_image[i][j]
            if value < 0:
                value = 0
            if value > 255:
                value = 255
            edge_crispening_image[i][j] = value
    return edge_crispening_image


def hough_transform(image, n):

    # Sobel edge detect
    grad_image = gradient_image(image, 'SOBEL')
    edge_image = threshld_image(grad_image, 150)

    # Generate sin and cos look-up table
    sin_tab = np.zeros(180)
    cos_tab = np.zeros(180)

    for angle in range(180):
        theta = angle*3.14159265358979/180.0
        cos_tab[angle] = np.cos(theta)
        sin_tab[angle] = np.sin(theta)

    # compute hough_img
    feature_point = 255
    row, col = edge_image.shape
    rmax = int(math.hypot(row, col))
    hough_image = np.zeros((180, rmax*2))

    for i in range(row):
        for j in range(col):
            if edge_image[i][j] == feature_point:
                for angle in range(180):                # for each angle
                    p = i*sin_tab[angle]+j*cos_tab[angle]  # compute p
                    p = int(p)                            # shift r to positive value
                    hough_image[angle][p+rmax] += 1       # accumulation
    plt.imshow(hough_image, cmap='gray')

    # draw top n line
    a = hough_image.reshape(1, -1)[0]
    index = np.argpartition(a, -n)[-n:]
    for idx in index:
        angle, p = np.unravel_index(idx, hough_image.shape)
        p -= rmax

        a = sin_tab[angle]
        b = cos_tab[angle]

        # 注意除以0的case
        if b < 1e-12:
            x0 = col
            y0 = (p-x0*b)/a

            x1 = 0
            y1 = (p-x1*b)/a
        else:
            y0 = row
            x0 = (p-y0*a)/b

            y1 = 0
            x1 = (p-y1*a)/b

        x0 = int(x0)
        y0 = int(y0)
        x1 = int(x1)
        y1 = int(y1)

        cv2.line(image, (x0, y0), (x1, y1), (255), 2)
    return image
