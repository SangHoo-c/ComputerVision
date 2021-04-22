import cv2
import numpy as np
import math
import time
from matplotlib import pyplot as plt


def add_padding(img, kernel):
    # read image
    img_col, img_row = img.shape
    kernel_col, kernel_row = kernel.shape
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
    img_col, img_row = image.shape
    kernel_col, kernel_row = kernel.shape

    pad = (kernel_row - 1) // 2

    # add padding
    image = add_padding(image, kernel)

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


def cross_correlation_1d(img, kernel):
    img_col = np.size(img, 0)
    img_width = np.size(img, 1)

    # convert 2d img to 1d
    tmp = img.flatten()

    # original size of 1d img
    img_1d_len = np.size(tmp, 0)
    c_c_lena_ve = np.array([tmp])

    # output after filtering
    result = np.zeros((img_1d_len, 1))

    if np.size(kernel, 1) != 1:
        print("horizontal")
        c_c_lena_ho = img.flatten()

        kernel_size = np.size(kernel, 1)
        add_len = np.size(kernel, 1) // 2
        # add first and last
        for i in range(add_len):
            c_c_lena_ho = np.insert(c_c_lena_ho, i, 0)
            c_c_lena_ho = np.append(c_c_lena_ho, 0)

        # transpose the array.
        c_c_lena_ho = np.array([c_c_lena_ho])

        # this is a main part of cross correlation
        for i in range(img_1d_len):
            result[i] = (kernel * c_c_lena_ho[0, i: i + kernel_size]).sum()

        result = np.reshape(result, (img_col, img_width))

        return result

    if np.size(kernel, 0) != 1:

        # 원래의 크기를 지키이 위해서
        # kernel 의 크기에 따라서 1d array 의 크기를 조정하고, cross correlation 을 수행

        print("vertical")

        kernel_size = np.size(kernel, 0)
        add_len = np.size(kernel, 0) // 2
        # add first and last
        for i in range(add_len):
            c_c_lena_ve = np.insert(c_c_lena_ve, i, 0)
            c_c_lena_ve = np.append(c_c_lena_ve, 0)

        # transpose the array.
        c_c_lena_ve = np.array([c_c_lena_ve])
        kernel = kernel.T

        # this is a main part of cross correlation
        for i in range(img_1d_len):
            result[i] = (kernel * c_c_lena_ve[0, i: i + kernel_size]).sum()

        result = np.reshape(result, (img_col, img_width))

        return result


# 1-2)
# a, b
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


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


def _return_result_for_assignment_1_question_d(input_img):
    if input_img == 'lenna':
        img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
    elif input_img == 'shapes':
        img = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
    else:
        print("false input... check your input")
        return -1

    # make image list
    im_1 = cross_correlation_2d(img, get_gaussian_filter_2d(5, 1))
    im_2 = cross_correlation_2d(img, get_gaussian_filter_2d(5, 6))
    im_3 = cross_correlation_2d(img, get_gaussian_filter_2d(5, 11))

    im_4 = cross_correlation_2d(img, get_gaussian_filter_2d(11, 1))
    im_5 = cross_correlation_2d(img, get_gaussian_filter_2d(11, 6))
    im_6 = cross_correlation_2d(img, get_gaussian_filter_2d(11, 11))

    im_7 = cross_correlation_2d(img, get_gaussian_filter_2d(17, 1))
    im_8 = cross_correlation_2d(img, get_gaussian_filter_2d(17, 6))
    im_9 = cross_correlation_2d(img, get_gaussian_filter_2d(17, 11))

    im1_s = cv2.resize(im_1, dsize=(0, 0), fx=0.5, fy=0.5)
    im2_s = cv2.resize(im_2, dsize=(0, 0), fx=0.5, fy=0.5)
    im3_s = cv2.resize(im_3, dsize=(0, 0), fx=0.5, fy=0.5)

    im4_s = cv2.resize(im_4, dsize=(0, 0), fx=0.5, fy=0.5)
    im5_s = cv2.resize(im_5, dsize=(0, 0), fx=0.5, fy=0.5)
    im6_s = cv2.resize(im_6, dsize=(0, 0), fx=0.5, fy=0.5)

    im7_s = cv2.resize(im_7, dsize=(0, 0), fx=0.5, fy=0.5)
    im8_s = cv2.resize(im_8, dsize=(0, 0), fx=0.5, fy=0.5)
    im9_s = cv2.resize(im_9, dsize=(0, 0), fx=0.5, fy=0.5)

    text_position = (10, 50)
    cv2.putText(im1_s,
                "5x5 s=1",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im2_s,
                "5x5 s=6",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im3_s,
                "5x5 s=11",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im4_s,
                "11x11 s=1",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im5_s,
                "11x11 s=6",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im6_s,
                "11x11 s=11",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im7_s,
                "17x17 s=1",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im8_s,
                "17x17 s=6",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)
    cv2.putText(im9_s,
                "17x17 s=11",
                text_position,
                cv2.QT_FONT_NORMAL,
                1,
                (209, 80, 0, 255),
                3)

    im_tile = concat_tile([[im1_s, im2_s, im3_s],
                           [im4_s, im5_s, im6_s],
                           [im7_s, im8_s, im9_s]]
                          )
    cv2.imwrite('./result/part_1_gaussian_filtered_' + input_img + '.png', im_tile)
    print("image write complete!")

    return 0


######################## Answers ##############################################

###############################################################################
# 2-3)
# c
# print(get_gaussian_filter_1d(5, 1))
# print(get_gaussian_filter_2d(5, 1))

###############################################################################
# d, f
_return_result_for_assignment_1_question_d("lenna")
_return_result_for_assignment_1_question_d("shapes")

###################################################################################
# e _ lenna
img_input_for_e = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

start_2d = time.time()
img_2d = cross_correlation_2d(img_input_for_e, get_gaussian_filter_2d(17, 6))
print("problem e) : time for lenna 2d is ", time.time() - start_2d)

start_1d = time.time()
# vertical
img_1d = cross_correlation_1d(img_input_for_e, get_gaussian_filter_1d(17, 6).reshape(-1, 1))
# horizontal
img_1d = cross_correlation_1d(img_1d, get_gaussian_filter_1d(17, 6).reshape(-1, 1).T)

print("problem e) : time for lenna 1d is ", time.time() - start_1d)

result_for_e = img_2d - img_1d
absolute_diff = 0

for i in range(result_for_e.shape[0]):
    for j in range(result_for_e.shape[1]):
        if abs(result_for_e[i, j]) != 255:
            absolute_diff += abs(result_for_e[i, j])
print("problem e) : sum of (absolute) lenna intensity differences", absolute_diff)

# 이거 visualize 하면 된다.
plt.imshow(img_2d, cmap='gray')
plt.title("2d conv")
plt.show()

plt.imshow(img_1d, cmap='gray')
plt.title("1d conv")
plt.show()

plt.imshow(img_2d - img_1d, cmap='gray')
plt.title("intensity differences.. between lenna 1d & 2d")
plt.show()
###############################################################################
# e _ shapes
img_input_for_e = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)

start_2d = time.time()
img_2d = cross_correlation_2d(img_input_for_e, get_gaussian_filter_2d(17, 6))
print("problem e) : time for shapes 2d is ", time.time() - start_2d)

start_1d = time.time()
img_1d = cross_correlation_1d(img_input_for_e, get_gaussian_filter_1d(17, 6).reshape(-1, 1))
img_1d = cross_correlation_1d(img_1d, get_gaussian_filter_1d(17, 6).reshape(-1, 1).T)
print("problem e) : time for shapes 1d is ", time.time() - start_1d)

result_for_e = img_2d - img_1d
absolute_diff = 0

for i in range(result_for_e.shape[0]):
    for j in range(result_for_e.shape[1]):
        if abs(result_for_e[i, j]) != 255:
            absolute_diff += abs(result_for_e[i, j])
print("problem e) : sum of (absolute) shapes intensity differences", absolute_diff)

# 이거 visualize 하면 된다.
plt.imshow(img_2d - img_1d, cmap='gray')
plt.title("intensity differences.. between shapes 1d & 2d")
plt.show()
