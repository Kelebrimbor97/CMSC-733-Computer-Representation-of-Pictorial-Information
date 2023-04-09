import cv2
from cv2 import fillPoly
import numpy as np
from scipy import interpolate


def get_bb(triangle_pts):
    xmin = np.amin(triangle_pts[0,:])
    xmax = np.amax(triangle_pts[0,:])
    ymin = np.amin(triangle_pts[1,:])
    ymax = np.amax(triangle_pts[1,:])
    return xmin, ymin, xmax, ymax

def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

def create_mask(target, features_list_target):
    mask = np.zeros((target.shape[0],target.shape[1]), np.uint8)
    points = np.array(features_list_target, np.int32)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    return mask

def blending(target_img, source_img, mask):
    radius = 3
    kernel = np.ones((radius, radius), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    return cv2.seamlessClone(
        target_img, source_img, mask, center, cv2.NORMAL_CLONE)

def draw_delaunay(i, triangleList, delaunay_color ) :
    img = i.copy()
    size = img.shape
    r = (0, 0, size[1], size[0])
    for t in triangleList :
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, 0)

    return img

def interpolate_source(i):
    image = i.copy()
    xs = np.linspace(0, image.shape[1], num=image.shape[1], endpoint=False)
    ys = np.linspace(0, image.shape[0], num=image.shape[0], endpoint=False)

    b = image[:, :, 0]
    interpolated_b = interpolate.interp2d(xs, ys, b, kind='cubic')

    g = image[:, :, 1]
    interpolated_g = interpolate.interp2d(xs, ys, g, kind='cubic')

    r = image[:, :, 2]
    interpolated_r = interpolate.interp2d(xs, ys, r, kind='cubic')

    return interpolated_b, interpolated_g, interpolated_r