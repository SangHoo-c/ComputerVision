import cv2

import numpy as np
import math
import time
from matplotlib import pyplot as plt

img = cv2.imread('./CV_Assignment_2_Images/smile.png', cv2.IMREAD_GRAYSCALE)


# 1-1 function.
def get_transformed_img(img_src, M):
    plane = np.zeros((801, 801), dtype=np.float32)
    plane += 255

    cv2.arrowedLine(plane, (0, 400), (801, 400), (0, 0, 0), 1, tipLength=0.03)
    cv2.arrowedLine(plane, (400, 801), (400, 0), (0, 0, 0), 1, tipLength=0.03)

    row, col = img_src.shape

    row_half = row // 2
    col_half = col // 2

    c_x = 400
    c_y = 400

    vec_hom_co = np.array([c_x - 400, c_y - 400, 1])
    c_x, c_y, k_ = M.dot(vec_hom_co)

    c_x = round(c_x)
    c_y = round(c_y)

    c_x += 400
    c_y += 400
    print(c_x, c_y)

    pos_x = c_x - row_half
    pos_y = c_y - col_half

    for i in range(c_x - row_half, c_x + row_half + 1):
        for j in range(c_y - col_half, c_y + col_half + 1):
            vec_hom_co = np.array([i - c_x, j - c_y, 1])
            i_, j_, k_ = M.dot(vec_hom_co)

            i_ = round(i_)
            j_ = (round(j_))

            i_ += c_x
            j_ += c_y

            if img_src[i - pos_x][j - pos_y] < 255:
                try:
                    plane[i_][j_] = img_src[i - pos_x][j - pos_y]
                except:
                    print("IndexError, plz go back to your boundary!")

    # plt.imshow(plane, cmap='gray')
    # plt.title("hi")
    # plt.show()

    # cv2.imshow("figure", plane)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return plane


e_m = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])


# 1-2 interactive keyboard event handling
def interactive_input():
    a_m = np.array([
        [1, 0, 0],
        [0, 1, -5],
        [0, 0, 1]
    ])

    d_m = np.array([
        [1, 0, 0],
        [0, 1, 5],
        [0, 0, 1]
    ])

    w_m = np.array([
        [1, 0, -5],
        [0, 1, 0],
        [0, 0, 1]
    ])

    s_m = np.array([
        [1, 0, 5],
        [0, 1, 0],
        [0, 0, 1]
    ])

    r_m = np.array([
        [math.cos(5 * math.pi / 180), -math.sin(5 * math.pi / 180), 0],
        [math.sin(5 * math.pi / 180), math.cos(5 * math.pi / 180), 0],
        [0, 0, 1]
    ])

    R_m = np.array([
        [math.cos(355 * math.pi / 180), -math.sin(355 * math.pi / 180), 0],
        [math.sin(355 * math.pi / 180), math.cos(355 * math.pi / 180), 0],
        [0, 0, 1]
    ])

    f_m = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])

    F_m = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    x_m = np.array([
        [1, 0, 0],
        [0, 0.95, 0],
        [0, 0, 1]
    ])

    X_m = np.array([
        [1, 0, 0],
        [0, 1.05, 0],
        [0, 0, 1]
    ])

    y_m = np.array([
        [0.95, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    Y_m = np.array([
        [1.05, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    tmp_m = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    while True:
        key = cv2.waitKey()
        h_flag = 1
        if key == ord('a'):
            tmp_m = a_m @ tmp_m
        elif key == ord('d'):
            tmp_m = d_m @ tmp_m
        elif key == ord('w'):
            tmp_m = w_m @ tmp_m
        elif key == ord('s'):
            tmp_m = s_m @ tmp_m
        elif key == ord('r'):
            tmp_m = r_m @ tmp_m
        elif key == ord('R'):
            tmp_m = R_m @ tmp_m
        elif key == ord('f'):
            tmp_m = f_m @ tmp_m
        elif key == ord('F'):
            tmp_m = F_m @ tmp_m
        elif key == ord('x'):
            tmp_m = x_m @ tmp_m
        elif key == ord('X'):
            tmp_m = X_m @ tmp_m
        elif key == ord('y'):
            tmp_m = y_m @ tmp_m
        elif key == ord('Y'):
            tmp_m = Y_m @ tmp_m
        elif key == ord('H'):
            h_flag = 0
        elif key == ord('Q'):
            break
        if h_flag == 1:
            img_ = get_transformed_img(img, tmp_m)
        else:
            img_ = get_transformed_img(img, e_m)
            h_flag = 1;
            tmp_m = e_m
        cv2.imshow("circle", img_)

    cv2.destroyAllWindows()


interactive_input()
