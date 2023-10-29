import math
import numpy as np
import cv2

def distance(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) **(1/2)

def HP(F, D0, n):
    idealHP = idealFilterHP(F, D0)*F
    gaussionHP = gaussionFilterHP(F, D0)*F
    butterworthHP = butterworthFilterHP(F, D0, n)*F

    # plt.figure(figsize=(6*3, 5*2), constrained_layout=False)
    # plt.subplot(231), plt.imshow(np.log(1 + np.abs(idealHP)), "gray"), plt.title("Ideal HP Filter")
    # plt.subplot(232), plt.imshow(np.log(1 + np.abs(gaussionHP)), "gray"), plt.title("Gaussian HP Filter")
    # plt.subplot(233), plt.imshow(np.log(1 + np.abs(butterworthHP)), "gray"), plt.title("Butterworth HP Filter")

    idealHP_output = np.fft.ifft2(np.fft.ifftshift(idealHP))
    gaussionHP_output = np.fft.ifft2(np.fft.ifftshift(gaussionHP))
    butterworthHP_output = np.fft.ifft2(np.fft.ifftshift(butterworthHP))

    # plt.subplot(234), plt.imshow(np.abs(idealHP_output), "gray")
    # plt.subplot(235), plt.imshow(np.abs(gaussionHP_output), "gray")
    # plt.subplot(236), plt.imshow(np.abs(butterworthHP_output), "gray")

    # plt.savefig("HP.png", format = 'png')

    return gaussionHP_output


def DFT(img):
    # plt.figure(figsize=(6*3, 5), constrained_layout=False)
    spectrum = np.fft.fft2(img)
    center_spectrum = np.fft.fftshift(spectrum)

    # plt.subplot(131), plt.imshow(img, "gray"), plt.title("Original")
    # plt.subplot(132), plt.imshow(np.log(1 + np.abs(spectrum)), "gray"), plt.title("Spectrum")
    # plt.subplot(133), plt.imshow(np.log(1 + np.abs(center_spectrum)), "gray"), plt.title("Centered spectrum")
    # plt.savefig('FFT.png', format='png')

    return center_spectrum


############### High-Pass Filter ##################
def idealFilterHP(F, D0):
    rows = len(F)
    cols = len(F[0])
    center = [rows/2, cols/2]
    H = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if(distance([i, j], center) > D0):
                H[i, j] = 1
    return H

def gaussionFilterHP(F, D0):
    rows = len(F)
    cols = len(F[0])
    center = [rows/2, cols/2]
    H = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            H[i, j] = 1 - math.exp((-distance([i, j], center)**2 / (2*(D0**2))))
    return H

def butterworthFilterHP(F, D0, n):
    rows = len(F)
    cols = len(F[0])
    center = [rows/2, cols/2]
    H = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            H[i, j] = 1 - 1/(1 + (distance([i, j], center)/D0)**(2*n))
    return H


############### Low-Pass Filter ##################
def idealFilterLP(F, D0):
    rows = len(F)
    cols = len(F[0])
    center = [rows/2, cols/2]
    H = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            if(distance([i, j], center) <= D0):
                H[i, j] = 1
    return H

def gaussionFilterLP(F, D0):
    rows = len(F)
    cols = len(F[0])
    center = [rows/2, cols/2]
    H = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            H[i, j] = math.exp((-distance([i, j], center)**2 / (2*(D0**2))))
    return H

def butterworthFilterLP(F, D0, n):
    rows = len(F)
    cols = len(F[0])
    center = [rows/2, cols/2]
    H = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            H[i, j] = 1/(1 + (distance([i, j], center)/D0)**(2*n))
    return H

def LP(F, D0, n):
    idealLP = idealFilterLP(F, D0)*F
    gaussionLP = gaussionFilterLP(F, D0)*F
    butterworthLP = butterworthFilterLP(F, D0, n)*F

    # plt.figure(figsize=(6*3, 5*2), constrained_layout=False)
    # plt.subplot(231), plt.imshow(np.log(1 + np.abs(idealLP)), "gray"), plt.title("Ideal LP Filter")
    # plt.subplot(232), plt.imshow(np.log(1 + np.abs(gaussionLP)), "gray"), plt.title("Gaussian LP Filter")
    # plt.subplot(233), plt.imshow(np.log(1 + np.abs(butterworthLP)), "gray"), plt.title("Butterworth LP Filter")

    idealLP_output = np.fft.ifft2(np.fft.ifftshift(idealLP))
    gaussionLP_output = np.fft.ifft2(np.fft.ifftshift(gaussionLP))
    butterworthLP_output = np.fft.ifft2(np.fft.ifftshift(butterworthLP))

    # plt.subplot(234), plt.imshow(np.abs(idealLP_output), "gray")
    # plt.subplot(235), plt.imshow(np.abs(gaussionLP_output), "gray")
    # plt.subplot(236), plt.imshow(np.abs(butterworthLP_output), "gray")

    # plt.savefig("LP.png", format = 'png')

    return idealLP_output
