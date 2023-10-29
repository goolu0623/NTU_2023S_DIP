import numpy as np
def on_mask_hit(image, mask):
    # mask目前卡死必須要是3*3 不然for loop那邊的邊界會壞掉
    i_row, i_col = image.shape
    m_row, m_col = len(mask), len(mask[0])
    result = np.zeros((i_row, i_col), np.uint8)
    for ir in range(1, i_row-1):
        for ic in range(1, i_col-1):
            flag = True
            for mr in range(m_row):
                for mc in range(m_col):
                    if mask[mr][mc] == image[ir+mr-1][ic+mc-1]:
                        continue
                    else:
                        flag = False
            if flag :
                result[ir][ic] = 255

    return result


def dilation(image, kernel):
    dil_image = np.zeros(image.shape, np.uint8)
    m, n = image.shape
    for i in range(m):
        for j in range(n):
            if image[i][j] > 0:
                for position in kernel:
                    p, q = position
                    if 0 <= (i + p) <= (m - 1) and 0 <= (j + q) <= (n - 1):
                        dil_image[i + p][j + q] = 255
    return dil_image


def erosion(image, kernel):
    ero_image = np.zeros(image.shape, np.uint8)
    rows, columns = ero_image.shape
    for i in range(1,rows-1):
        for j in range(1,columns-1):
            flag = True
            for position in kernel:
                p, q = position
                if 0 <= (i + p) < rows and 0 <= (j + q) < columns and image[i + p, j + q] == 0:
                    flag = False
                    break
            if flag:
                ero_image[i][j] = 255
    return ero_image

def opening(image, kernel):
    return dilation(erosion(image, kernel), kernel)


def closing(image, kernel):
    return erosion(dilation(image, kernel), kernel)


def convolution(F, kernel_filter): # in same place
    img = F.copy()
    kernel_size = kernel_filter.shape[0]
    gap = kernel_size // 2
    pad_img = np.lib.pad(img,(gap,gap),'edge')
    result_img = np.zeros((pad_img.shape))
    
    height, width = pad_img.shape
    for i in range(gap, height - gap):
        for j in range(gap, width - gap):
            patch = pad_img[i-gap:i+gap+1,j-gap:j+gap+1] * kernel_filter
            result_img[i][j] = patch.sum()

    return result_img[gap:height-gap,gap:width-gap] # clip padding


# k-means algorithm: https://gist.github.com/tvwerkhoven/4fdc9baad760240741a09292901d3abd
def kMeans(X, K, iterations):
    # Select k vectors as the initial centroids
    centroids = X[np.random.choice(np.arange(len(X)), K)]
    for i in range(iterations):
        # 找出與centroids距離最相近的向量(相減平方最小)
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # check if there are fewer than K clusters.(如果centroids有重複就有可能發生)
        if(len(np.unique(C)) < K):
            centroids = X[np.random.choice(np.arange(len(X)), K)]
        else: #以平均向量作為中心點
            centroids = [X[C == k].mean(axis = 0) for k in range(K)]
    return C


