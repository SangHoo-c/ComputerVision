from custom_function import *

img_lenna_input = read_image("lenna")
img_shapes_input = read_image("shapes")

# 2-1) a.
img_lenna = cross_correlation_2d(img_lenna_input, get_gaussian_filter_2d(7, 1.5))
img_shapes = cross_correlation_2d(img_shapes_input, get_gaussian_filter_2d(7, 1.5))


# 2-2)
def compute_image_gradient(img, image_name):
    start_compute_gradient = time.time()

    img_col = img.shape[0]
    img_row = img.shape[1]

    kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # a)
    edge_x = cross_correlation_2d(img, kernel_x)
    edge_y = cross_correlation_2d(img, kernel_y)

    # b)
    # compute magnitude and direction of gradient for each pixel
    magnitude_sobel = np.sqrt(np.square(edge_x) + np.square(edge_y))
    magnitude_sobel *= 255.0 / magnitude_sobel.max()

    # same result with for for 문
    magnitude_sobel = np.hypot(edge_x, edge_y)
    direction_sobel = np.arctan(np.divide(edge_x, edge_y, out=np.zeros_like(edge_x), where=edge_y != 0))

    # d) computation time
    print("2-2. problem d) : time for computing " + image_name + " gradient is ", time.time() - start_compute_gradient)

    plt.imshow(magnitude_sobel, cmap='gray')
    plt.title("image gradient")
    plt.show()
    cv2.imwrite('./result/part_2_edge_raw_' + image_name + '.png', magnitude_sobel)

    return magnitude_sobel, direction_sobel


# 2-3)
def non_maximum_suppression_dir(mag, dir):
    start_NMS = time.time()

    image_col = mag.shape[0]
    image_row = mag.shape[1]
    revised_mag = np.zeros((image_col, image_row))

    if image_row != image_col:
        image_name = "shapes"
    else:
        image_name = "lenna"

    for i in range(image_col):
        for j in range(image_row):
            if dir[i, j] < 0:
                dir[i, j] += 360

            if ((j + 1) < image_row) and ((j - 1) >= 0) and ((i + 1) < image_col) and ((i - 1) >= 0):
                # 0 degrees
                if dir[i, j] >= 337.5 or dir[i, j] < 22.5:
                    if mag[i, j] >= mag[i, j + 1] and mag[i, j] >= mag[i, j - 1]:
                        revised_mag[i, j] = mag[i, j]
                # 45 degrees
                if 22.5 <= dir[i, j] < 67.5:
                    if mag[i, j] >= mag[i - 1, j + 1] and mag[i, j] >= mag[i + 1, j - 1]:
                        revised_mag[i, j] = mag[i, j]
                # 90 degrees
                if 67.5 <= dir[i, j] < 112.5:
                    if mag[i, j] >= mag[i - 1, j] and mag[i, j] >= mag[i + 1, j]:
                        revised_mag[i, j] = mag[i, j]
                # 135 degrees
                if 112.5 <= dir[i, j] < 157.5:
                    if mag[i, j] >= mag[i - 1, j - 1] and mag[i, j] >= mag[i + 1, j + 1]:
                        revised_mag[i, j] = mag[i, j]
                # 180 degrees
                if 157.5 <= dir[i, j] < 202.5:
                    if mag[i, j] >= mag[i, j + 1] and mag[i, j] >= mag[i, j - 1]:
                        revised_mag[i, j] = mag[i, j]
                # 225 degrees
                if 202.5 <= dir[i, j] < 247.5:
                    if mag[i, j] >= mag[i - 1, j + 1] and mag[i, j] >= mag[i + 1, j - 1]:
                        revised_mag[i, j] = mag[i, j]
                # 270 degrees
                if 247.5 <= dir[i, j] < 292.5:
                    if mag[i, j] >= mag[i - 1, j] and mag[i, j] >= mag[i + 1, j]:
                        revised_mag[i, j] = mag[i, j]
                # 315 degrees
                if 292.5 <= dir[i, j] < 337.5:
                    if mag[i, j] >= mag[i - 1, j - 1] and mag[i, j] >= mag[i + 1, j + 1]:
                        revised_mag[i, j] = mag[i, j]

    print("2-3. problem d) : time for " + image_name + "'s computing NMS is ", time.time() - start_NMS)

    plt.imshow(revised_mag, cmap='gray')
    plt.title("magnitude after NMS")
    plt.show()
    cv2.imwrite('./result/part_2_edge_sup_' + image_name + '.png', revised_mag)
    return 0


(lenna_magnitude, lenna_direction) = compute_image_gradient(img_lenna, "lenna")
(shapes_magnitude, shapes_direction) = compute_image_gradient(img_shapes, "shapes")

non_maximum_suppression_dir(lenna_magnitude, lenna_direction)
non_maximum_suppression_dir(shapes_magnitude, shapes_direction)
