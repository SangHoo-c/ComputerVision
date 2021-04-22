from image_header import *


# 2-1
def bfmMatcher(src1, src2):
    orb = cv2.ORB_create()

    kp1 = orb.detect(src1, None)
    kp1, des1 = orb.compute(src1, kp1)

    kp2 = orb.detect(src2, None)
    kp2, des2 = orb.compute(src2, kp2)

    matches = []
    for i in range(len(des1)):
        tmp_min = 1000
        tmp_queryIdx = i
        tmp_trainIdx = 0
        for j in range(len(des2)):
            dist = hamming_distance(des1[i], des2[j])
            # dist = cv2.norm(des1[i], des2[j], cv2.NORM_HAMMING)
            if tmp_min > dist:
                tmp_trainIdx = j
                tmp_min = dist
        match = cv2.DMatch(_queryIdx=tmp_queryIdx, _trainIdx=tmp_trainIdx, _imgIdx=i, _distance=tmp_min)
        matches.append(match)

    # 이 부분을 구현해보자
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(des1, des2)

    # list 자료형 matches
    # matches 인스턴스의 attribute 인 distance 길이를 기준으로 정렬
    matches = sorted(matches, key=lambda x: x.distance)
    for i in range(10):
        print("---------")
        print(i)
        print("번째")
        print(matches[i].imgIdx)
        print(matches[i].queryIdx)
        print(matches[i].trainIdx)
        print(matches[i].distance)

    img_output = cv2.drawMatches(src1, kp1, src2, kp2, matches[:10], None, flags=2)

    cv2.imshow("figure", img_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 2-2
# 2-3
def compute_homography_ransac(src, dst, thresh):
    corr = np.zeros((src.shape[0], src.shape[1] * 2))
    for i in range(src.shape[0]):
        corr[i, 0] = src[i, 0]
        corr[i, 1] = src[i, 1]
        corr[i, 2] = dst[i, 0]
        corr[i, 3] = dst[i, 1]

    maxInliers = []
    finalH = None
    src = np.zeros((4, 2))
    dst = np.zeros((4, 2))

    start = time.time()

    for i in range(1000):
        end = time.time()
        print(end - start)
        if end - start > 3:
            break

        # find 4 random points to calculate a homography
        corr1 = corr[random.randrange(0, len(corr))]
        corr2 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((corr1, corr2))
        corr3 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr3))
        corr4 = corr[random.randrange(0, len(corr))]
        randomFour = np.vstack((randomFour, corr4))

        # call the homography function on those points
        for i in range(4):
            src[i, 0] = randomFour[i][0]
            src[i, 1] = randomFour[i][1]
            dst[i, 0] = randomFour[i][2]
            dst[i, 1] = randomFour[i][3]

        # print(src)
        # print(dst)

        h = compute_homography(src, dst)
        # print(h)
        inliers = []

        for i in range(len(corr)):
            d = geometricDistance(corr[i, 0:2], corr[i, 2:4], h)
            # d = geometricDistance(src[i], dst[i], h)
            # print(corrs[i, 0:2])
            # print(corrs[i, 2:4])
            # print(d)
            if d < 5:
                inliers.append(corr[i])

        if len(inliers) > len(maxInliers):
            maxInliers = inliers
            finalH = h

        if len(maxInliers) > (len(corr) * thresh):
            break

        return finalH


def result_2_4():
    src, dst = find_x_y_cor(imageA, imageB)
    src2, dst2 = find_x_y_cor(img_hp_cover, imageB)

    ransac_H_, status = cv2.findHomography(src, dst, cv2.RANSAC, 1.0)
    ransac_H_2, status = cv2.findHomography(src2, dst2, cv2.RANSAC, 1.0)
    ransac_H = compute_homography_ransac(src, dst, 1)
    ransac_H2 = compute_homography_ransac(src2, dst2, 1)
    normal_H = compute_homography(src, dst)

    result1 = cv2.warpPerspective(imageA, normal_H, (imageB.shape[1], imageB.shape[0]))
    result2 = cv2.warpPerspective(imageA, normal_H, (imageB.shape[1], imageB.shape[0]))
    result3 = cv2.warpPerspective(imageA, ransac_H_, (imageB.shape[1], imageB.shape[0]))
    result4 = cv2.warpPerspective(imageA, ransac_H_, (imageB.shape[1], imageB.shape[0]))
    result5 = cv2.warpPerspective(img_hp_cover, ransac_H_2, (imageB.shape[1], imageB.shape[0]))

    # 검은 배경에 이미지 넣기
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            if result4[i, j] < 10:
                result4[i, j] = imageB[i, j]
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            if result2[i, j] < 10:
                result2[i, j] = imageB[i, j]
    for i in range(imageB.shape[0]):
        for j in range(imageB.shape[1]):
            if result5[i, j] < 10:
                result5[i, j] = imageB[i, j]

    cv2.imshow('result1', result1)
    cv2.imshow('result2', result2)
    cv2.imshow('result3', result3)
    cv2.imshow('result4', result4)
    cv2.imshow('result5', result5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 2-5
def image_stitching(imageA, imageB):
    orb = cv2.ORB_create()

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


    src, dst = find_x_y_cor(imageA, imageB)

    # findHomography 함수를 구현하자.
    # 2-3
    # H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    H = compute_homography(src, dst)

    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0: imageA.shape[0], 0: imageB.shape[1]] = imageB

    # image blending
    # dst = result[0:imageB.shape[0], imageB.shape[1] - 5: imageB.shape[1] + 5]
    for i in range(10):
        tmp1 = result[0:imageB.shape[0], imageB.shape[1] - i]
        tmp2 = result[0:imageB.shape[0], imageB.shape[1] + i]
        result[0:imageB.shape[0], imageB.shape[1] - i: imageB.shape[1] - i + 1 ] = cv2.addWeighted(tmp1, 0.7, tmp2, 0.3, 0)
        result[0:imageB.shape[0], imageB.shape[1] + i: imageB.shape[1] + i + 1] = cv2.addWeighted(tmp1, 0.3, tmp2, 0.7, 0)


    cv2.imshow('blur_result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # 2-1
    # bfmMatcher(img_cover, img_desk)

    # 2-3
    # result_2_4()

    # 2-5
    # image_stitching(img_dia_11, img_dia_10)

    return 0


if __name__ == "__main__":
    main()
