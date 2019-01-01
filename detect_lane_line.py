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

# from courves import Curves


# curves = Curves(number_of_windows = 9, margin = 100, minimum_pixels = 50,
#                 ym_per_pix = 30 / 720 , xm_per_pix = 3.7 / 700)


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

    # img = birds_eye(image)
    #input image must be Warped Image

    s_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]

    l_channel = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)[:, :, 0]

    b_channel = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)[:, :, 2]

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
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8'))

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


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low),
        #               (win_xleft_high, win_y_high), (0, 255, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low),
        #               (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    if len(leftx) == 0 or len(lefty) == 0:
        return out_img

    # # Fit a second order polynomial to each using `np.polyfit`
    # left_fit = np.polyfit(lefty, leftx, 2)
    # right_fit = np.polyfit(righty, rightx, 2)
    #
    # # Generate x and y values for plotting
    # ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    #
    # try:
    #
    #     left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    #     right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # except TypeError:
    #     # Avoids an error if `left` and `right_fit` are still none or incorrect
    #     print('The function failed to fit a line!')
    #     left_fitx = 1 * ploty ** 2 + 1 * ploty
    #     right_fitx = 1 * ploty ** 2 + 1 * ploty
    #
    # ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    #
    # left_xs = left_fit[0] * (ploty ** 2) + left_fit[1] * ploty + left_fit[2]
    # right_xs = right_fit[0] * (ploty ** 2) + right_fit[1] * ploty + right_fit[2]
    # xls, xrs, ys = left_xs.astype(np.uint32), right_xs.astype(np.uint32), ploty.astype(np.uint32)

    # # int(t / 2)
    # for xl, xr, y in zip(xls, xrs, ys):
    #     cv2.line(out_img, (xl, y), (xl, y), (255, 203, 57), 4)
    #     cv2.line(out_img, (xr, y), (xr, y), (255, 203, 57), 4)

    return out_img


def image_process_pipeline(img):
    offset = [0, 320, 640, 960]
    width, height = 320, 180

    result_img = img

    brideye_img = birds_eye(img)

    result = curves.fit(brideye_img)
    left_curve = result['pixel_left_best_fit_curve']
    right_curve = result['pixel_right_best_fit_curve']

    left_radius = result['left_radius']
    right_radius = result['right_radius']
    pos = result['vehicle_position_words']
    curve_debug_img = result['image']

    # resize_birdeye_img = cv2.resize(brideye_img, dsize=(0, 0), fx=0.25, fy=0.25)
    #
    # combined_binary_img = apply_thresholds(brideye_img, show=False)
    # debug_img = fit_polynomial(combined_binary_img)
    # resize_debug_img = cv2.resize(debug_img, dsize=(0, 0), fx=0.25, fy=0.25)
    #
    #
    # result_img[:height, offset[0]: offset[0] + width] = resize_birdeye_img
    # result_img[:height, offset[1]: offset[1] + width] = resize_debug_img

    # cv2.putText(img, 'wokao', (20, 220), fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    print(pos)
    return brideye_img


def process_video(input_video_file):
    clip1 = VideoFileClip(input_video_file)
    outputclip = clip1.fl_image(image_process_pipeline)
    outputclip.write_videofile('output_'+input_video_file, audio=False);


if __name__ == "__main__":
    print('main')
    # orig_img = cv2.imread('test_images/test2.jpg')
    # # undist_img = undistort(orig_img)
    # brid_eye_img = birds_eye(orig_img)
    # combined_binary_img = apply_thresholds(brid_eye_img, show=False)
    # out_img = fit_polynomial(combined_binary_img)
    #
    #
    #
    # f, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(9, 6))
    # ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    # ax1.set_title('Original', fontsize=20)
    # ax2.imshow(cv2.cvtColor(brid_eye_img, cv2.COLOR_BGR2RGB))
    # ax2.set_title('Bird eye', fontsize=20)
    #
    # ax3.imshow(combined_binary_img, cmap='gray')
    # ax3.set_title('Combined binary', fontsize=20)
    #
    #
    # ax4.imshow(out_img)
    #
    # plt.show()
    # plt.savefig('undistort2.png')

    # orig_img = cv2.imread('test_images/test2.jpg')
    # processed_img = image_process_pipeline(orig_img)
    # plt.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    # plt.show()

    # process_video('project_video.mp4')

