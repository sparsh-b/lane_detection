import os
import copy
import cv2
import numpy as np
import math

vid_path = 'whiteline.mp4'
vidObj = cv2.VideoCapture(vid_path)
success = 1
frames = []
rgb_frames = []

while success:
    success, rgb_image = vidObj.read()
    try:
        image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    except:
        assert len(frames) != 0, 'No frames found!!! Check video path'
        break
    frames.append(image)
    rgb_frames.append(rgb_image)

def line_from_points(gp1_pts):
    point1 = [gp1_pts[0][0], -gp1_pts[0][1]]
    point2 = [gp1_pts[1][0], -gp1_pts[1][1]]
    a = point1[1] - point2[1]
    b = point2[0] - point1[0]
    c = a*(point2[0]) + b*(point2[1])
    return a, b, c

def find_bottom_point(a, b, c, h):
    y = -(h-1)
    x = (c-b*y)/a
    return x, -y

def find_intersection(a1, b1, c1, a2, b2, c2):
    x = (b2*c1 - b1*c2) / (a1*b2 - a2*b1)
    y = (c2 - a2*x)/b2
    y *= -1
    return x, y

def imshow_components(orig_labels, unique_label):
    labels = copy.copy(orig_labels)
    labels[labels!=unique_label] = 0
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    cv2.imshow('label-{}'.format(unique_label), labeled_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def count_white_points_on_line(gp1_line, ref_bloated):
    bloated = copy.copy(ref_bloated)
    num_white_gp1 = 0
    for point in gp1_line:
        point = np.around(point, 0).astype(int)
        if point[0] < 0 or point[1] < 0 or point[0] >= w or point[1] >= h:
            continue
        else:
            if bloated[point[1], point[0]] == 255:
                num_white_gp1 += 1
    return num_white_gp1

for frame_idx in range(len(frames)):
    img = frames[frame_idx]
    rgb_img = rgb_frames[frame_idx]
    _, bin_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((3,3), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel2, iterations=1)
    bin_img = cv2.erode(bin_img, kernel2, iterations=1)
    (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(bin_img, 8, cv2.CV_32S)
    values, counts = np.unique(labels, return_counts=True)
    num_ignore_labels = 0
    consider_labels = []
    cluster_size_threshold_low = 50
    cluster_size_threshold_high = 80000
    for i in range(counts.shape[0]):#filter clusters based on size
        if (counts[i] < cluster_size_threshold_low) or (counts[i] > cluster_size_threshold_high):# or (stats[values[i], cv2.CC_STAT_AREA] < 200):
            num_ignore_labels += 1
            labels[labels == values[i]] = 0
        else:
            consider_labels.append(values[i])

    labels[labels != 0] = 1
    bin_img = (labels * 255).astype(np.uint8)
    h, w = bin_img.shape
    lines_img = np.zeros((h, w, 3)).astype(np.uint8)
    low = 20
    high = 50
    bloated = copy.copy(bin_img)
    gray_rgb_img = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    gray_rgb_img = cv2.erode(gray_rgb_img, kernel2, iterations=1)
    bin_img = cv2.Canny(image=bin_img, threshold1=low, threshold2=high, L2gradient=True)
    minLineLength = 2
    maxLineGap = 100
    lines = cv2.HoughLines(bin_img, rho=1,theta=np.pi/180, threshold=20, min_theta=0, max_theta=np.pi)

    if lines is not None:
        gp1_r = []
        gp1_t = []
        gp2_r = []
        gp2_t = []
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]

            if abs(theta-math.radians(50)) < math.radians(10):#not (theta >= 0 and theta <= np.pi/4):
                gp1_r.append(rho)
                gp1_t.append(theta)
            elif abs(theta-math.radians(120)) < math.radians(10):
                gp2_r.append(rho)
                gp2_t.append(theta)
    
        gp = 0
        for (rho, theta) in [(np.mean(gp1_r), np.mean(gp1_t)), (np.mean(gp2_r), np.mean(gp2_t))]:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            if gp == 0:
                gp1_pts = [pt1, pt2]
            else:
                gp2_pts = [pt1, pt2]
            gp += 1

    a1, b1, c1 = line_from_points(gp1_pts)
    a2, b2, c2 = line_from_points(gp2_pts)
    x1, y1 = find_bottom_point(a1, b1, c1, h)
    x2, y2 = find_bottom_point(a2, b2, c2, h)
    x3, y3 = find_intersection(a1, b1, c1, a2, b2, c2)
    triangle_pts = np.array([[x1, y1], [x2, y2], [x3, y3]]).astype(int)
    cv2.drawContours(rgb_img, [triangle_pts], 0, (50,0,50), -1)

    num_steps=1000
    gp1_x_inc = (x1 - x3) / num_steps
    gp1_y_inc = (y1 - y3) / num_steps
    gp2_x_inc = (x2 - x3) / num_steps
    gp2_y_inc = (y2 - y3) / num_steps
    gp1_line = [(x3 + i * gp1_x_inc, y3 + i * gp1_y_inc) for i in range(num_steps)]
    gp2_line = [(x3 + i * gp2_x_inc, y3 + i * gp2_y_inc) for i in range(num_steps)]

    dividers = copy.copy(gray_rgb_img)
    bloated = cv2.dilate(bloated, kernel2, iterations=2)
    num_white_gp1 = count_white_points_on_line(gp1_line, bloated)
    num_white_gp2 = count_white_points_on_line(gp2_line, bloated)
    
    if num_white_gp1 < num_white_gp2:
        cv2.line(gray_rgb_img, (int(x1), int(y1)), (int(x3), int(y3)), (0,0,255), 2, cv2.LINE_AA)
        cv2.line(gray_rgb_img, (int(x2), int(y2)), (int(x3), int(y3)), (0,255,0), 2, cv2.LINE_AA)
    else:
        cv2.line(gray_rgb_img, (int(x2), int(y2)), (int(x3), int(y3)), (0,0,255), 2, cv2.LINE_AA)
        cv2.line(gray_rgb_img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2, cv2.LINE_AA)

    bottom_imgs = cv2.hconcat([dividers, gray_rgb_img])
    bottom_imgs = cv2.resize(bottom_imgs, dsize=(int(bottom_imgs.shape[1]/2), bottom_imgs.shape[0]))
    top_img = cv2.vconcat([rgb_img, bottom_imgs])
    top_img = cv2.resize(top_img, dsize=(int(top_img.shape[1]*0.8), int(top_img.shape[0]*0.8)))
    cv2.imshow('output', top_img)
    if not os.path.exists('./results'):
        os.makedirs('./results')
    cv2.imwrite('results/lanes{}.jpg'.format(str(frame_idx).zfill(3)), top_img)
    cv2.waitKey(5)

print('Lanes detected!!!')