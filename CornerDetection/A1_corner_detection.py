from custom_function import *

img_lenna_input = read_image("lenna")
img_shapes_input = read_image("shapes")

img_lenna = cross_correlation_2d(img_lenna_input, get_gaussian_filter_2d(7, 1.5))
img_shapes = cross_correlation_2d(img_shapes_input, get_gaussian_filter_2d(7, 1.5))


def compute_corner_response(img):
    start_corner_detections = time.time()

    image_col = img.shape[0]
    image_row = img.shape[1]

    if image_row != image_col:
        image_name = "shapes"
        # gray scale 2차원의 이미지를 3차원으로 변경한다.
        rgb_stacked_img = np.stack((img_shapes_input,) * 3, axis=-1)

    else:
        image_name = "lenna"
        # gray scale 2차원의 이미지를 3차원으로 변경한다.
        rgb_stacked_img = np.stack((img_lenna_input,) * 3, axis=-1)

    # a)
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    edge_x = cross_correlation_2d(img, kernel_x)
    edge_y = cross_correlation_2d(img, kernel_y)

    IxIx = edge_x * edge_x
    # IxIy , IyIx 같은 값을 갖는다.
    IxIy = edge_x * edge_y
    IyIy = edge_y * edge_y

    window_size = 5
    padding = window_size // 2

    r_arr = np.zeros((image_col, image_row))

    for i in range(padding, image_col - padding):
        for j in range(padding, image_row - padding):
            w_IxIx = IxIx[i - padding:i + padding + 1, j - padding:j + padding + 1]
            w_IyIy = IyIy[i - padding:i + padding + 1, j - padding:j + padding + 1]
            w_IxIy = IxIy[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # b)
            # get a second moment matrix
            M_a = w_IxIx.sum()
            M_d = w_IyIy.sum()
            M_b_c = w_IxIy.sum()

            det_M = M_a * M_d - M_b_c * M_b_c
            trace_M = M_a + M_d

            # c)
            r_arr[i, j] = det_M - 0.04 * (trace_M ** 2)

    # d)
    # change negative value to 0
    # normalize value [0, 1]

    r_arr = np.where(r_arr < 0, 0, r_arr)
    r_max = np.amax(r_arr)
    r_min = np.amin(r_arr)

    r_arr = (r_arr - r_min) / (r_max - r_min)
    # to visualize multiply 255 to normalized value.
    r_arr *= 255

    # e)
    plt.imshow(r_arr, cmap='gray')
    plt.title("corner detection")
    plt.show()
    cv2.imwrite('./result/part_3_corner_raw_' + image_name + '.png', r_arr)

    print("2-3. problem d) : time for corner detection for " + image_name + " is : ",
          time.time() - start_corner_detections)

    # 3-3
    # a)

    # r_arr 를 다시 정규화 시키기위해 0 ~ 1 범위로 돌린다.
    r_arr /= 255

    for i in range(padding, image_col - padding):
        for j in range(padding, image_row - padding):
            if r_arr[i, j] > 0.1:
                rgb_stacked_img.itemset((i, j, 0), 0)
                rgb_stacked_img.itemset((i, j, 1), 255)
                rgb_stacked_img.itemset((i, j, 2), 0)

    # b)
    plt.imshow(rgb_stacked_img, cmap='gray')
    plt.title("corner detection after NMS")
    plt.show()
    cv2.imwrite('./result/part_3_corner_bin_' + image_name + '.png', rgb_stacked_img)

    return 0


def non_maximum_suppression_win(R, winSize, img):
    start_corner_detections_NMS_window = time.time()

    image_col = img.shape[0]
    image_row = img.shape[1]

    if image_row != image_col:
        image_name = "shapes"
        # gray scale 2차원의 이미지를 3차원으로 변경한다.
        rgb_stacked_img = np.stack((img_shapes_input,) * 3, axis=-1)

    else:
        image_name = "lenna"
        # gray scale 2차원의 이미지를 3차원으로 변경한다.
        rgb_stacked_img = np.stack((img_lenna_input,) * 3, axis=-1)

    # a)
    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    edge_x = cross_correlation_2d(img, kernel_x)
    edge_y = cross_correlation_2d(img, kernel_y)

    IxIx = edge_x * edge_x
    # IxIy , IyIx 같은 값을 갖는다.
    IxIy = edge_x * edge_y
    IyIy = edge_y * edge_y

    window_size = winSize
    padding = window_size // 2

    r_arr = np.zeros((image_col, image_row))

    for i in range(padding, image_col - padding):
        for j in range(padding, image_row - padding):
            w_IxIx = IxIx[i - padding:i + padding + 1, j - padding:j + padding + 1]
            w_IyIy = IyIy[i - padding:i + padding + 1, j - padding:j + padding + 1]
            w_IxIy = IxIy[i - padding:i + padding + 1, j - padding:j + padding + 1]

            # b)
            # get a second moment matrix
            M_a = w_IxIx.sum()
            M_d = w_IyIy.sum()
            M_b_c = w_IxIy.sum()

            det_M = M_a * M_d - M_b_c * M_b_c
            trace_M = M_a + M_d

            # c)
            r_arr[i, j] = det_M - 0.04 * (trace_M ** 2)

    r_arr = np.where(r_arr < 0, 0, r_arr)
    r_max = np.amax(r_arr)
    r_min = np.amin(r_arr)

    r_arr = (r_arr - r_min) / (r_max - r_min)

    for i in range(padding, image_col - padding):
        for j in range(padding, image_row - padding):

            # local maximum 을 구하는 과정
            if r_arr[i, j] > R:
                tmp_max = np.max(r_arr[i - padding: i + padding + 1, j - padding: j + padding + 1])
                if r_arr[i, j] >= tmp_max:
                    # i, j 순서 중요
                    cv2.circle(rgb_stacked_img, (j, i), 5, (0, 255, 0), 2)
                rgb_stacked_img.itemset((i, j, 0), 0)
                # rgb_stacked_img.itemset((i, j, 1), 255)
                rgb_stacked_img.itemset((i, j, 2), 0)

    # d )
    plt.imshow(rgb_stacked_img, cmap='gray')
    plt.title("corner detection after custom window NMS")
    plt.show()
    cv2.imwrite('./result/part_3_corner_sup_' + image_name + '.png', rgb_stacked_img)

    print("3-3. problem d) : time for custom NMS corner detection is : ",
          time.time() - start_corner_detections_NMS_window)
    return 0


compute_corner_response(img_shapes)
compute_corner_response(img_lenna)


non_maximum_suppression_win(0.1, 11, img_shapes)
non_maximum_suppression_win(0.1, 11, img_lenna)

