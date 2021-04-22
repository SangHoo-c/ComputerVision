import cv2

import numpy as np
import math
import time
from matplotlib import pyplot as plt


def read_image(input_img):
    if input_img == 'lenna':
        img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    elif input_img == 'shapes':
        img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    else:
        print("false input... check your input")
        return -1
    return img


def add_padding(img, kernel):
    # read image
    (img_col, img_row) = img.shape[:2]
    (kernel_col, kernel_row) = kernel.shape[:2]
    pad = (kernel_row - 1) // 2

    # create new image of desired size and color (blue) for padding
    ww = img_row + 2 * pad  # row
    hh = img_col + 2 * pad  # column

    color = 0
    result = np.full((hh, ww), color)

    # compute center offset
    j_row = (ww - img_row) // 2
    i_col = (hh - img_col) // 2

    # copy img image into center of result image
    result[i_col:i_col + img_col, j_row:j_row + img_row] = img

    return result


def cross_correlation_2d(image, kernel):
    (img_col, img_row) = image.shape[:2]
    (kernel_col, kernel_row) = kernel.shape[:2]

    pad = (kernel_row - 1) // 2

    # add padding
    # 문제가 있는 부분.

    # corner detection 을 제외한 경우
    # add_padding 함수를 정의해서 사용했습니다.
    # filtering 과 edge detection 에 전혀 문제가 없었습니다.
    # 하지만, corner detection 의 경우, 결과값이 제대로 나오질 않아서, 무엇이 문제일까 고민하다가
    # openCV 의 내장 함수를 사용해보니 제대로 결과값이 나왔습니다.
    # 이런 상황이었다는 점, 확인부탁드립니다.

    # custom padding 함수
    # image = add_padding(image, kernel)

    # openCV 함수
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # default
    output = np.zeros((img_col, img_row))

    # each (i, j)-coordinate from left-to-right and top to
    # bottom
    for i in np.arange(pad, img_col + pad):
        for j in np.arange(pad, img_row + pad):
            # i, j 부분을 pad 만큼 추출한다.
            roi = image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            # cross correlation 을 수행한다.
            k = (roi * kernel).sum()
            output[i - pad, j - pad] = k

    return output


def get_gaussian_filter_1d(size, sigma):
    kernel = np.ones(size)
    size_ = size // 2
    density = np.array(range(-size_, size_ + 1))
    sum_ = 0

    for i in range(size):
        # 가우시안 필터 공식 적용
        kernel[i] = float(math.exp(-float(density[i] ** 2) / (float(2) * (sigma ** 2))) /
                          float(math.sqrt((2 * math.pi * sigma ** 2))))
        sum_ += kernel[i]
    return kernel / sum_


def get_gaussian_filter_2d(size, sigma):
    kernel_1d = get_gaussian_filter_1d(size, sigma)
    kernel_2d = np.outer(kernel_1d, kernel_1d.T)
    return kernel_2d


