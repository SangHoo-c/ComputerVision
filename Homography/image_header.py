import math
import cv2
import numpy as np
import time
import random

# img_cover = cv2.imread('./CV_Assignment_2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
# img_desk = cv2.imread('./CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
imageA = cv2.imread('./CV_Assignment_2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)  # 책 사진
imageB = cv2.imread('./CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)  # 책상 사진

img_hp_cover = cv2.imread('./CV_Assignment_2_Images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
img_hp_cover = cv2.resize(img_hp_cover, (imageA.shape[1], imageA.shape[0]))


# 오른쪽 사진
img_dia_11 = cv2.imread('./CV_Assignment_2_Images/diamondhead-11.png', cv2.IMREAD_GRAYSCALE)

# 왼쪽 사진
img_dia_10 = cv2.imread('./CV_Assignment_2_Images/diamondhead-10.png', cv2.IMREAD_GRAYSCALE)


def find_x_y_cor(src1, src2):
    size_n = 35

    m_a = np.zeros((size_n, 2))
    m_b = np.zeros((size_n, 2))
    orb = cv2.ORB_create()

    kp1 = orb.detect(src1, None)
    kp1, des1 = orb.compute(src1, kp1)

    kp2 = orb.detect(src2, None)
    kp2, des2 = orb.compute(src2, kp2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    dic = {}
    dic_list = []

    # list 자료형 matches
    # matches 인스턴스의 attribute 인 distance 길이를 기준으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)
    for i in range(size_n):
        dic = {'src': matches[i].queryIdx, 'des': matches[i].trainIdx}
        dic_list.append(dic)
    i = 0
    for dic in dic_list:
        # print("src 의 x, y 좌표 : ")
        # print(kp1[dic['src']].pt[0])
        # print(kp1[dic['src']].pt[1])

        m_a[i][0] = kp1[dic['src']].pt[0]
        m_a[i][1] = kp1[dic['src']].pt[1]
        # print(m_a[i][0])

        # print("des 의 x, y 좌표 : ")
        # print(kp2[dic['des']].pt[0])
        # print(kp2[dic['des']].pt[1])

        m_b[i][0] = kp2[dic['des']].pt[0]
        m_b[i][1] = kp2[dic['des']].pt[1]

        # print(m_b[i][0])
        i += 1
    return m_a, m_b


def hamming_distance(des1, des2):
    dis = 0
    for i in range(len(des1)):
        dis += bin(des1[i] ^ des2[i]).count('1')
        # cmp1 = '{0:08b}'.format(des1[i])
        # cmp2 = '{0:08b}'.format(des2[i])
    return dis


def geometricDistance(np1, np2, h):
    p1 = np.array([
        [np1[0], np1[1], 1]
    ])
    p2 = np.array([
        [np2[0], np2[1], 1]
    ])
    des_p1 = np.dot(h, p1.T)
    des_p1 = des_p1 / des_p1[-1]

    error = des_p1.T - p2

    return np.linalg.norm(error)


# 2-2
def compute_homography(srcP, destP):
    size_n = srcP.shape[0]

    trans_src_x, trans_src_y = np.mean(srcP, axis=0)
    trans_dest_x, trans_dest_y = np.mean(destP, axis=0)

    trans_src_M = np.array([
        [1, 0, -trans_src_x],
        [0, 1, -trans_src_y],
        [0, 0, 1]
    ])

    trans_dest_M = np.array([
        [1, 0, -trans_dest_x],
        [0, 1, -trans_dest_y],
        [0, 0, 1]
    ])

    tmp_src = np.zeros((size_n * 2, 2))
    tmp_dst = np.zeros((size_n * 2, 2))

    # 2-2 b)1) 원점 (0, 0)
    for i in range(size_n):
        x_before = np.array([srcP[i][0], srcP[i][1], 1])
        x_after = trans_src_M @ x_before
        tmp_src[i][0] = x_after[0]
        tmp_src[i][1] = x_after[1]

    for i in range(size_n):
        x_before = np.array([destP[i][0], destP[i][1], 1])
        x_after = trans_dest_M @ x_before
        tmp_dst[i][0] = x_after[0]
        tmp_dst[i][1] = x_after[1]

    # 2-2 b)2) 반지름 root 2 인 원안에 모든 점을 넣기
    # 길이를 구해서 그걸 기준으로 둘다 같은 값으로 shrink 해야함
    src_nor_dist = np.zeros((size_n, 1))
    dest_nor_dist = np.zeros((size_n, 1))

    for i in range(size_n):
        src_nor_dist[i] = math.sqrt(tmp_src[i][0] ** 2 + tmp_src[i][1] ** 2)
        dest_nor_dist[i] = math.sqrt(tmp_dst[i][0] ** 2 + tmp_dst[i][1] ** 2)

    shrink_src = np.max(src_nor_dist)
    shrink_dest = np.max(dest_nor_dist)

    shrink_src_M = np.array([
        [math.sqrt(2) / shrink_src, 0, 0],
        [0, math.sqrt(2) / shrink_src, 0],
        [0, 0, 1]
    ])

    shrink_dest_M = np.array([
        [math.sqrt(2) / shrink_dest, 0, 0],
        [0, math.sqrt(2) / shrink_dest, 0],
        [0, 0, 1]
    ])

    T_s = shrink_src_M @ trans_src_M
    T_d = shrink_dest_M @ trans_dest_M

    # T_s , T_d 변환matrix 정의
    for i in range(size_n):
        x_before = np.array([[srcP[i][0], srcP[i][1], 1]]).T
        x_after = T_s @ x_before
        srcP[i][0] = x_after[0]
        srcP[i][1] = x_after[1]

    for i in range(size_n):
        x_before = np.array([[destP[i][0], destP[i][1], 1]]).T
        x_after = T_d @ x_before
        destP[i][0] = x_after[0]
        destP[i][1] = x_after[1]

    # 표준화된 srcP, desP
    nor_src = srcP
    nor_des = destP

    # 2-2 d)
    A = []
    for i in range(size_n):
        x, y = nor_src[i][0], nor_src[i][1]
        u, v = nor_des[i][0], nor_des[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)

    # 최종적인 homography matrix 구하기
    out_put_h_n = np.dot(np.linalg.inv(T_d), np.dot(H, T_s))
    print(out_put_h_n)

    return out_put_h_n


orb = cv2.ORB_create(
    # nfeatures=1000
)

kpA = orb.detect(imageA, None)
kpA, desA = orb.compute(imageA, kpA)

kpB = orb.detect(imageB, None)
kpB, desB = orb.compute(imageB, kpB)

matches = []
for i in range(len(desA)):
    tmp_min = 1000
    tmp_queryIdx = i
    tmp_trainIdx = 0
    for j in range(len(desB)):
        dist = hamming_distance(desA[i], desB[j])
        # dist = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
        if tmp_min > dist:
            tmp_trainIdx = j
            tmp_min = dist
    match = cv2.DMatch(_queryIdx=tmp_queryIdx, _trainIdx=tmp_trainIdx, _imgIdx=i, _distance=tmp_min)
    matches.append(match)

sorted_matches = sorted(matches, key=lambda x: x.distance)
