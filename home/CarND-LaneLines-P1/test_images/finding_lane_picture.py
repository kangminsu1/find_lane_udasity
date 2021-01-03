import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

directory = os.listdir('./find_lane')
files = []

for i in range(len(directory)):
    file_name = directory[i].split('.')
    if file_name[1] == 'jpg':
        files.append(directory[i])

for number in range(len(files)):
    image = mpimg.imread('./find_lane/' + files[number])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    ignore_mask_color = 255

    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20

    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    mask = np.zeros_like(edges)


    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    line_image = np.copy(image) * 0

    Lanes = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    for lane in Lanes:
        for x1, y1, x2, y2 in lane:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

    color_edges = np.dstack((edges, edges, edges))

    edges_lines = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

    plt.imshow(edges_lines)
    plt.show()

