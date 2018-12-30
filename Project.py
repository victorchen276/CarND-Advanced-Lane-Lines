import math
import os, glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle

import random
import os, sys
from moviepy.editor import VideoFileClip
from collections import deque

import re


def draw_images(images, cmap=None, savefile=False):
    cols = 2
    rows = (len(images) + 1) // cols

    fig = plt.figure(figsize=(20, 10))
    # fig = plt.figure(figsize=(6.4, 4.8))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)

    if savefile == True:
        plt.savefig('test_images_output/output.png')
        plt.close(fig)
    else:
        plt.show()


def hist(img):
    cp_img = img.copy()
    cp_img = cp_img/255
    bottom_half = cp_img[cp_img.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def camera_calibration():
    x_cor = 9  # Number of corners to find
    y_cor = 6
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((y_cor * x_cor, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('camera_cal/calibration*.jpg')
    images = sorted(images, key=lambda x: float(re.findall("(\d+)", x)[0]))


    # Step through the list and search for chessboard corners
    corners_not_found = []  # Calibration images in which opencv failed to find corners
    plt.figure(figsize=(12, 18))  # Figure for calibration images
    # plt.figtext(0.5, 0.9, 'Image with corners patterns drawn', fontsize=22, ha='center')
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (x_cor, y_cor), None)  # Find the chessboard corners
        # If found, add object points, image points
        if ret is True:
            objpoints.append(objp)
            imgpoints.append(corners)
            plt.subplot(5, 4, len(imgpoints))

            cv2.drawChessboardCorners(img, (x_cor, y_cor), corners, ret)
            plt.imshow(img)
            plt.title(fname)
            plt.axis('off')
        else:
            corners_not_found.append(fname)
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
    # plt.savefig('success.png')

    print('Corners were found on', str(len(imgpoints)), 'out of', str(len(images)), 'it is', str(
        len(imgpoints) * 100.0 / len(images)), '% of calibration images')
    # Draw pictures
    plt.figure(figsize=(12, 4))
    # plt.figtext(.5, .8, 'Images in which cv2 failed to find desired corners', fontsize=22, ha='center')
    for i, p in enumerate(corners_not_found):
        plt.subplot(1, 3, i + 1)
        plt.imshow(mpimg.imread(p))  # draw the first image of each class
        plt.title(p)
        plt.axis('off')
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()
    # plt.savefig('fail.png')

    img1 = cv2.imread(images[0])
    h, w = img1.shape[:2]
    ret, camera_matrix, distortion_coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    calibration_data = {
        "camera_matrix": camera_matrix,
        "distortion_coefficient": distortion_coef
    }

    pickle.dump(calibration_data, open("camera_calibration_data.p", "wb"))


def undistort(img):
    calibration_data = pickle.load(open("camera_calibration_data.p", "rb"))
    camera_matrix = calibration_data['camera_matrix']
    dist_coef = calibration_data['distortion_coefficient']
    undist = cv2.undistort(img, camera_matrix, dist_coef, None, camera_matrix)
    return undist


# Perform perspective transform
def birds_eye(image):
    undist = undistort(image)
    img_size = (undist.shape[1], undist.shape[0])
    src = np.float32([(580, 460), #top left
                      (200, 720), #bottom left
                      (1110, 720), #bottom right
                      (703, 460)]) #top right

    dst = np.float32([(320, 0), #top left
                      (320, 720), #bottom left
                      (960, 720),
                      (960, 0)])


    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)
    return warped


# Create binary thresholded images to isolate lane line pixels
def apply_thresholds(image, show=True):

    img = birds_eye(image)
    # img = image

    s_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:, :, 2]

    l_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)[:, :, 0]

    b_channel = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, 2]

    # Threshold color channel
    s_thresh_min = 180
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    b_thresh_min = 155
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 225
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # color_binary = np.dstack((u_binary, s_binary, l_binary))

    combined_binary = np.zeros_like(s_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    if show == True:
        # Plotting thresholded images
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharey='col', sharex='row', figsize=(10, 4))
        f.tight_layout()

        ax1.set_title('Original Image', fontsize=16)
        ax1.imshow(cv2.cvtColor(undistort(image), cv2.COLOR_BGR2RGB))

        ax2.set_title('Warped Image', fontsize=16)
        ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'))

        ax3.set_title('s binary threshold', fontsize=16)
        ax3.imshow(s_binary, cmap='gray')

        ax4.set_title('b binary threshold', fontsize=16)
        ax4.imshow(b_binary, cmap='gray')

        ax5.set_title('l binary threshold', fontsize=16)
        ax5.imshow(l_binary, cmap='gray')

        ax6.set_title('Combined color thresholds', fontsize=16)
        ax6.imshow(combined_binary, cmap='gray')
        plt.show()
    else:
        return combined_binary


def fill_lane(image):
    # print(image_path)
    # img = cv2.copy
    #     image.copy()
    combined_binary = apply_thresholds(image, show=False)

    rightx = []
    righty = []
    leftx = []
    lefty = []

    x, y = np.nonzero(np.transpose(combined_binary))
    i = 720
    j = 630
    while j >= 0:
        histogram = np.sum(combined_binary[j:i, :], axis=0)
        left_peak = np.argmax(histogram[:640])
        x_idx = np.where((((left_peak - 25) < x) & (x < (left_peak + 25)) & ((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            leftx.extend(x_window.tolist())
            lefty.extend(y_window.tolist())

        right_peak = np.argmax(histogram[640:]) + 640
        x_idx = np.where((((right_peak - 25) < x) & (x < (right_peak + 25)) & ((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            rightx.extend(x_window.tolist())
            righty.extend(y_window.tolist())
        i -= 90
        j -= 90

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + right_fit[2]
    rightx_int = right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2]
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_fit[0] * 0 ** 2 + right_fit[1] * 0 + right_fit[2])
    righty = np.append(righty, 0)
    leftx_int = left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2]
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_fit[0] * 0 ** 2 + left_fit[1] * 0 + left_fit[2])
    lefty = np.append(lefty, 0)
    lsort = np.argsort(lefty)
    rsort = np.argsort(righty)
    lefty = lefty[lsort]
    leftx = leftx[lsort]
    righty = righty[rsort]
    rightx = rightx[rsort]
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + right_fit[2]

    # Measure Radius of Curvature for each lane line
    ym_per_pix = 30. / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
    left_curverad = ((1 + (2 * left_fit_cr[0] * np.max(lefty) + left_fit_cr[1]) ** 2) ** 1.5) \
                    / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(lefty) + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])

    # Calculate the position of the vehicle
    center = abs(640 - ((rightx_int + leftx_int) / 2))

    offset = 0
    img_size = (image.shape[1], image.shape[0])
    src = np.float32([(580, 460),  # top left
                      (200, 720),  # bottom left
                      (1110, 720),  # bottom right
                      (703, 460)])  # top right

    dst = np.float32([(320, 0),  # top left
                      (320, 720),  # bottom left
                      (960, 720),
                      (960, 0)])
    Minv = cv2.getPerspectiveTransform(dst, src)

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, lefty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, righty]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0, 255), thickness=50)
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (combined_binary.shape[1], combined_binary.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)

    return result

    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    # f.tight_layout()
    # ax1.imshow(cv2.cvtColor((birds_eye(img)), cv2.COLOR_BGR2RGB))
    # ax1.set_xlim(0, 1280)
    # ax1.set_ylim(0, 720)
    # ax1.plot(left_fitx, lefty, color='red', linewidth=3)
    # ax1.plot(right_fitx, righty, color='red', linewidth=3)
    # ax1.set_title('Fit Polynomial to Lane Lines', fontsize=16)
    # ax1.invert_yaxis()  # to visualize as we do the images
    # ax2.imshow(result)
    #
    # ax2.set_title('Fill Lane Between Polynomials', fontsize=16)
    # if center < 640:
    #     ax2.text(200, 100, 'Vehicle is {:.2f}m left of center'.format(center * 3.7 / 700),
    #              style='italic', color='white', fontsize=10)
    # else:
    #     ax2.text(200, 100, 'Vehicle is {:.2f}m right of center'.format(center * 3.7 / 700),
    #              style='italic', color='white', fontsize=10)
    # ax2.text(200, 175, 'Radius of curvature is {}m'.format(int((left_curverad + right_curverad) / 2)),
    #          style='italic', color='white', fontsize=10)
    #
    # plt.show()


class Line:
    def __init__(self):
        # Was the line found in the previous frame?
        self.found = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0

    def found_search(self, x, y):
        '''
        This function is applied when the lane lines have been detected in the previous frame.
        It uses a sliding window to search for lane pixels in close proximity (+/- 25 pixels in the x direction)
        around the previous detected polynomial.
        '''
        xvals = []
        yvals = []
        if self.found == True:
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i, j])
                xval = (np.mean(self.fit0)) * yval ** 2 + (np.mean(self.fit1)) * yval + (np.mean(self.fit2))
                x_idx = np.where((((xval - 25) < x) & (x < (xval + 25)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) == 0:
            self.found = False  # If no lane pixels were detected then perform blind search
        return xvals, yvals, self.found

    def blind_search(self, x, y, image):
        '''
        This function is applied in the first few frames and/or if the lane was not successfully detected
        in the previous frame. It uses a slinding window approach to detect peaks in a histogram of the
        binary thresholded image. Pixels in close proimity to the detected peaks are considered to belong
        to the lane lines.
        '''
        xvals = []
        yvals = []
        if self.found == False:
            i = 720
            j = 630
            while j >= 0:
                histogram = np.sum(image[j:i, :], axis=0)
                if self == Right:
                    peak = np.argmax(histogram[640:]) + 640
                else:
                    peak = np.argmax(histogram[:640])
                x_idx = np.where((((peak - 25) < x) & (x < (peak + 25)) & ((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) > 0:
            self.found = True
        else:
            yvals = self.Y
            xvals = self.X
        return xvals, yvals, self.found

    def radius_of_curvature(self, xvals, yvals):
        ym_per_pix = 30. / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
        fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)
        curverad = ((1 + (2 * fit_cr[0] * np.max(yvals) + fit_cr[1]) ** 2) ** 1.5) \
                   / np.absolute(2 * fit_cr[0])
        return curverad

    def sort_vals(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0] * 720 ** 2 + polynomial[1] * 720 + polynomial[2]
        top = polynomial[0] * 0 ** 2 + polynomial[1] * 0 + polynomial[2]
        return bottom, top

Left = Line()
Right = Line()

objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

def process_vid(image):
    img_size = (image.shape[1], image.shape[0])

    # Calibrate camera and undistort image
    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    # undist = cv2.undistort(image, mtx, dist, None, mtx)

    undist = undistort(image)

    # Perform perspective transform
    offset = 0
    src = np.float32([[490, 482], [810, 482],
                      [1250, 720], [0, 720]])
    dst = np.float32([[0, 0], [1280, 0],
                      [1250, 720], [40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, img_size)

    # Generate binary thresholded images
    b_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2Lab)[:, :, 2]
    l_channel = cv2.cvtColor(warped, cv2.COLOR_RGB2LUV)[:, :, 0]

    # Set the upper and lower thresholds for the b channel
    b_thresh_min = 145
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    # Set the upper and lower thresholds for the l channel
    l_thresh_min = 215
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    # Identify all non zero pixels in the image
    x, y = np.nonzero(np.transpose(combined_binary))

    if Left.found == True:  # Search for left lane pixels around previous polynomial
        leftx, lefty, Left.found = Left.found_search(x, y)

    if Right.found == True:  # Search for right lane pixels around previous polynomial
        rightx, righty, Right.found = Right.found_search(x, y)

    if Right.found == False:  # Perform blind search for right lane lines
        rightx, righty, Right.found = Right.blind_search(x, y, combined_binary)

    if Left.found == False:  # Perform blind search for left lane lines
        leftx, lefty, Left.found = Left.blind_search(x, y, combined_binary)

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)

    # Calculate left polynomial fit based on detected pixels
    left_fit = np.polyfit(lefty, leftx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    leftx_int, left_top = Left.get_intercepts(left_fit)

    # Average intercepts across n frames
    Left.x_int.append(leftx_int)
    Left.top.append(left_top)
    leftx_int = np.mean(Left.x_int)
    left_top = np.mean(Left.top)
    Left.lastx_int = leftx_int
    Left.last_top = left_top

    # Add averaged intercepts to current x and y vals
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_top)
    lefty = np.append(lefty, 0)

    # Sort detected pixels based on the yvals
    leftx, lefty = Left.sort_vals(leftx, lefty)

    Left.X = leftx
    Left.Y = lefty

    # Recalculate polynomial with intercepts and average across n frames
    left_fit = np.polyfit(lefty, leftx, 2)
    Left.fit0.append(left_fit[0])
    Left.fit1.append(left_fit[1])
    Left.fit2.append(left_fit[2])
    left_fit = [np.mean(Left.fit0),
                np.mean(Left.fit1),
                np.mean(Left.fit2)]

    # Fit polynomial to detected pixels
    left_fitx = left_fit[0] * lefty ** 2 + left_fit[1] * lefty + left_fit[2]
    Left.fitx = left_fitx

    # Calculate right polynomial fit based on detected pixels
    right_fit = np.polyfit(righty, rightx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    rightx_int, right_top = Right.get_intercepts(right_fit)

    # Average intercepts across 5 frames
    Right.x_int.append(rightx_int)
    rightx_int = np.mean(Right.x_int)
    Right.top.append(right_top)
    right_top = np.mean(Right.top)
    Right.lastx_int = rightx_int
    Right.last_top = right_top
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_top)
    righty = np.append(righty, 0)

    # Sort right lane pixels
    rightx, righty = Right.sort_vals(rightx, righty)
    Right.X = rightx
    Right.Y = righty

    # Recalculate polynomial with intercepts and average across n frames
    right_fit = np.polyfit(righty, rightx, 2)
    Right.fit0.append(right_fit[0])
    Right.fit1.append(right_fit[1])
    Right.fit2.append(right_fit[2])
    right_fit = [np.mean(Right.fit0), np.mean(Right.fit1), np.mean(Right.fit2)]

    # Fit polynomial to detected pixels
    right_fitx = right_fit[0] * righty ** 2 + right_fit[1] * righty + right_fit[2]
    Right.fitx = right_fitx

    # Compute radius of curvature for each lane in meters
    left_curverad = Left.radius_of_curvature(leftx, lefty)
    right_curverad = Right.radius_of_curvature(rightx, righty)

    # Only print the radius of curvature every 3 frames for improved readability
    if Left.count % 3 == 0:
        Left.radius = left_curverad
        Right.radius = right_curverad

    # Calculate the vehicle position relative to the center of the lane
    position = (rightx_int + leftx_int) / 2
    distance_from_center = abs((640 - position) * 3.7 / 700)

    Minv = cv2.getPerspectiveTransform(dst, src)

    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left.fitx, Left.Y])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, Right.Y]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0, 0, 255), thickness=40)
    cv2.fillPoly(color_warp, np.int_(pts), (34, 255, 34))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)

    # Print distance from center on video
    if position > 640:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (100, 80),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (100, 80),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int((Left.radius + Right.radius) / 2)), (120, 140),
                fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    Left.count += 1
    return result

def image_process_pipeline(img):
    offset = [0, 320, 640, 960]
    width, height = 320, 180
    brideye_img = birds_eye(img)
    resize_birdeye_img = cv2.resize(brideye_img, dsize=(0, 0), fx=0.25, fy=0.25)
    result_img = fill_lane(img)
    result_img[:height, offset[0]: offset[0] + width] = resize_birdeye_img

    # cv2.putText(img, 'wokao', (20, 220), fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)

    return result_img

def process_video(input_video_file):
    clip1 = VideoFileClip(input_video_file);
    outputclip = clip1.fl_image(process_vid)
    outputclip.write_videofile('output_'+input_video_file, audio=False);

if __name__ == "__main__":
    print('main')
    # execute only if run as a script
    # camera_calibration()
    # undistort('camera_cal/calibration1.jpg')
    # undistort('test_images/test1.jpg')

    # orig_img = cv2.imread('test_images/test1.jpg')
    # undist_img = undistort(orig_img)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    # ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    # ax1.set_title('Original', fontsize=20)
    # ax2.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
    # ax2.set_title('Undistorted', fontsize=20)
    # # plt.show()
    # plt.savefig('undistort2.png')

    # i = 1
    # for image in glob.glob('test_images/test*.jpg'):
    #     orig_img = cv2.imread(image)
    #     birdeye_img = birds_eye(orig_img)
    #     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    #     f.tight_layout()
    #     ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    #     ax1.set_title('Original', fontsize=20)
    #     ax2.imshow(cv2.cvtColor(birdeye_img, cv2.COLOR_BGR2RGB))
    #     ax2.set_title('Undistorted and Warped Image', fontsize=20)
    #     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #     plt.show()
    #     # plt.savefig('warp_' + str(i) + '.png')
    #     i += 1

    # orig_img = cv2.imread('test_images/test1.jpg')
    # apply_thresholds(orig_img)

    # for image in glob.glob('test_images/test*.jpg'):
    #     orig_img = cv2.imread(image)
    #     apply_thresholds(orig_img)

    # for image in glob.glob('test_images/test*.jpg'):
    #     fill_lane(image)


    # video process
    # fill_lane('test_images/test2.jpg')

    # process_video('harder_challenge_video.mp4')
    # process_video('challenge_video.mp4')
    # process_video('project_video.mp4')

    # orig_img = cv2.imread('test_images/test2.jpg')
    # processed_img = image_process_pipeline(orig_img)
    # plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    # plt.show()

    # orig_img = cv2.imread('test_images/test2.jpg')
    # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 6))
    # ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    # ax1.set_title('Original', fontsize=20)
    # # ax2.imshow(cv2.cvtColor(undist_img, cv2.COLOR_BGR2RGB))
    #
    #
    # combined_binary = apply_thresholds(orig_img, show=False)
    #
    # img3 = np.zeros((combined_binary.shape[0], combined_binary.shape[1], 3))
    # img3[:, :, 0] = combined_binary
    # img3[:, :, 1] = combined_binary
    # img3[:, :, 2] = combined_binary
    # ax2.imshow(img3)
    # ax2.set_title('combined_binary', fontsize=20)
    #
    # bird_eye_img = birds_eye(orig_img)
    # # ax3.imshow(bird_eye_img, cv2.COLOR_BGR2RGB)
    # ax3.imshow(cv2.cvtColor(bird_eye_img, cv2.COLOR_BGR2RGB))
    # ax3.set_title('birdeye', fontsize=20)
    #
    # # histogram = hist(combined_binary)
    # # ax4.imshow(histogram)
    # # ax4.set_title('histogram', fontsize=20)
    #
    # # result = cv2.addWeighted(bird_eye_img, 1, img3, 0.5, 0)
    # # ax4.imshow(result)
    # # ax4.set_title('result', fontsize=20)
    #
    #
    # plt.show()

    # plt.savefig('undistort2.png')








