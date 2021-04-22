import cv2
import numpy as np
import math
import random
import time
from matplotlib import pyplot as plt
from skimage.measure import ransac
from compute_avg_reproj_error import *

img_temple1 = cv2.imread('./temple1.png')
img_temple2 = cv2.imread('./temple2.png')

img_h1 = cv2.imread('./house1.jpg')
img_h2 = cv2.imread('./house2.jpg')

img_lib1 =cv2.imread('./library1.jpg')
img_lib2 = cv2.imread('./library2.jpg')

M = np.loadtxt('temple_matches.txt')
M_hous = np.loadtxt('house_matches.txt')
M_library = np.loadtxt('library_matches.txt')


def compute_F_raw(Mat):
    # a)
    # fp1 = (x1, y1)
    # fp2 = (x2, y2)
    fp1 = M[:, 0:2]
    fp2 = M[:, 2:4]

    n = M.shape[0]

    # eight-point algorithm
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [fp1[i, 0] * fp2[i, 0], fp1[i, 0] * fp2[i, 1], fp1[i, 0],
                fp1[i, 1] * fp2[i, 0], fp1[i, 1] * fp2[i, 1], fp1[i, 1],
                fp2[i, 0], fp2[i, 1], 1]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    return F / F[2, 2]


def compute_F_norm(Mat):
    fp1 = M[:, 0:2]
    fp2 = M[:, 2:4]

    # normalize
    fp1 = fp1.T
    fp2 = fp2.T

    # 2차원의 좌표를 homogeneous 로 변경
    fp1 = np.insert(fp1, 2, 1, axis=0)
    fp2 = np.insert(fp2, 2, 1, axis=0)

    n = M.shape[0]

    # scaling value - to get a mean sqaured distance, I m gonna use a standard deviation.
    mean_fp1 = np.mean(fp1, axis=1)
    S_fp1 = math.sqrt(2) / np.std(fp1[:])
    T_fp1 = np.array([[S_fp1, 0, -S_fp1 * mean_fp1[0]], [0, S_fp1, -S_fp1 * mean_fp1[1]], [0, 0, 1]])

    fp1 = np.dot(T_fp1, fp1)

    mean_fp2 = np.mean(fp2, axis=1)
    S_fp2 = math.sqrt(2) / np.std(fp2[:])
    T_fp2 = np.array([[S_fp2, 0, -S_fp2 * mean_fp2[0]], [0, S_fp2, -S_fp2 * mean_fp2[1]], [0, 0, 1]])

    fp2 = np.dot(T_fp2, fp2)

    # eight-point algorithm
    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [fp1[0, i] * fp2[0, i], fp1[0, i] * fp2[1, i], fp1[0, i] * fp2[2, i],
                fp1[1, i] * fp2[0, i], fp1[1, i] * fp2[1, i], fp1[1, i] * fp2[2, i],
                fp1[2, i] * fp2[0, i], fp1[2, i] * fp2[1, i], fp1[2, i] * fp2[2, i]]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # rank-2 constraint on F
    U, S, V = np.linalg.svd(F)

    # 가장 작은 singular value 를 0 으로 만들어준다.
    S[2] = 0

    # 다시 되돌려준다.
    F = np.dot(U, np.dot(np.diag(S), V))

    # 이제 T_fp1, T_fp2 변환 matrix 를 이용해서 normalized 를 완성한다.
    F = np.dot(T_fp1.T, np.dot(F, T_fp2))

    return F / F[2, 2]


def compute_F_ransac(Mat):
    return 0


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1-draw a polar line on img1 at the point on img2
            lines-epipolar line'''
    print(img1.shape)
    r = img1.shape[0]
    c = img1.shape[1]

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
    return img1, img2


def draw_ep(M, img1, img2):
    fp1 = M[:, 0:2]
    fp2 = M[:, 2:4]

    # normalize
    fp1 = fp1.T
    fp2 = fp2.T

    # 2차원의 좌표를 homogeneous 로 변경
    fp1 = np.insert(fp1, 2, 1, axis=0)
    fp2 = np.insert(fp2, 2, 1, axis=0)

    F = compute_F_norm(M)


    # Draw a polar line in the left image for the points in the right image
    lines1 = cv2.computeCorrespondEpilines(fp2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, fp1, fp2)

    # In the left picture, draw the epipolar line in the right picture
    lines2 = cv2.computeCorrespondEpilines(fp1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, fp2, fp1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    # cv2.imshow("img1", img5)
    # cv2.imshow('img2', img3)
    # #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# b)
raw_F = compute_F_raw(M)
raw_F_h = compute_F_raw(M_hous)
raw_F_l = compute_F_raw(M_library)
# print(raw_F)

# c)
nor_F = compute_F_norm(M)
nor_F_h = compute_F_norm(M_hous)
nor_F_l = compute_F_norm(M_library)
# print(nor_F)

ran_F = compute_F_ransac(M)
ran_F_h = compute_F_ransac(M_hous)
ran_F_l = compute_F_ransac(M_library)

# e)
raw_error = compute_avg_reproj_error(M, raw_F)
raw_error_h = compute_avg_reproj_error(M_hous, raw_F_h)
raw_error_l = compute_avg_reproj_error(M_library, raw_F_l)

nor_error = compute_avg_reproj_error(M, nor_F)
nor_error_h = compute_avg_reproj_error(M_hous, nor_F_h)
nor_error_l = compute_avg_reproj_error(M_library, nor_F_l)

# ran_error = compute_avg_reproj_error(M, ran_F)
# ran_error_h = compute_avg_reproj_error(M_hous, ran_F_h)
# ran_error_l = compute_avg_reproj_error(M_library, ran_F_l)

print(" Average Reprojection Errors (temple1.png & temple2.png)")
print("  Raw = %.10f" % raw_error)
print("  Norm = %.10f" % nor_error)
# print("  Ransac = %.10f" % ran_error)

print(" Average Reprojection Errors (house1.jpg & house2.jpg)")
print("  Raw = %.10f" % raw_error_h)
print("  Norm = %.10f" % nor_error_h)
# print("  Ransac = %.10f" % ran_error_h)

print(" Average Reprojection Errors (library1.jpg & library2.jpg)")
print("  Raw = %.10f" % raw_error_l)
print("  Norm = %.10f" % nor_error_l)
# print("  Ransac = %.10f" % ran_error_l)


draw_ep(M, img_temple1, img_temple2)
draw_ep(M_hous, img_h1, img_h2)
draw_ep(M_library, img_lib1, img_lib2)


