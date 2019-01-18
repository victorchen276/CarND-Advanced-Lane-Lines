import numpy as np
import cv2

from source.camera import camera
from source.LaneLine import LaneLine
from source.window import Window


import matplotlib.pyplot as plt

class LaneDetect(object):

    def __init__(self, first_frame):
        self.num_windows = 9
        self.leftLane = None
        self.rightLane = None

        self.l_windows = []
        self.r_windows = []

        (self.h, self.w, _) = first_frame.shape
        self.camera = camera()

        self.load_camera()
        self.init_lines(first_frame)

    def load_camera(self, load_data=True, calibration_data='./camera_calibration_data.p', images=''):
        if load_data is True:
            self.camera.load_calibration_data(calibration_data)
        else:
            self.camera.calibration(images, x_cor=9, y_cor=6, outputfilename='./camera_calibration_data_1.p')

    def init_lines(self, frame):
        """
        Finds starting points for left and right lines (e.g. lane edges) and initialises Window and Line objects.
        Parameters
        ----------
        frame   : Frame to scan for lane edges.
        """
        # Take a histogram of the bottom half of the image
        # edges = get_edges(frame)
        # (flat_edges, _) = flatten_perspective(edges)

        # undist = self.camera_calibrate.undistort(frame)

        edges = self.get_edges(frame)
        flat_edges, unwarp_matrix = self.camera.birds_eye(edges)
        # birdeye_img, unwarp_matrix = self.camera_calibrate.birds_eye(frame)
        # flat_edges = get_edges(birdeye_img)
        histogram = np.sum(flat_edges[int(self.h / 2):, :], axis=0)

        nonzero = flat_edges.nonzero()
        # Create empty lists to receive left and right lane pixel indices
        l_indices = np.empty([0], dtype=np.int)
        r_indices = np.empty([0], dtype=np.int)
        window_height = int(self.h / self.num_windows)

        for i in range(self.num_windows):
            l_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.l_windows[-1].x if len(self.l_windows) > 0 else np.argmax(histogram[:self.w // 2])
            )
            r_window = Window(
                y1=self.h - (i + 1) * window_height,
                y2=self.h - i * window_height,
                x=self.r_windows[-1].x if len(self.r_windows) > 0 else np.argmax(histogram[self.w // 2:]) + self.w // 2
            )
            # Append nonzero indices in the window boundary to the lists
            l_indices = np.append(l_indices, l_window.pixels_in(nonzero), axis=0)
            r_indices = np.append(r_indices, r_window.pixels_in(nonzero), axis=0)
            self.l_windows.append(l_window)
            self.r_windows.append(r_window)
        self.leftLane = LaneLine(x=nonzero[1][l_indices], y=nonzero[0][l_indices], h=self.h, w=self.w)
        self.rightLane = LaneLine(x=nonzero[1][r_indices], y=nonzero[0][r_indices], h=self.h, w=self.w)


    def process_pipeline(self, img):

        output_img = img.copy()

        # birdeye_img, unwarp_matrix = self.camera_calibrate.birds_eye(img)
        # warped_edges_img = get_edges(birdeye_img)

        edges = self.get_edges(img)
        flat_edges, unwarp_matrix = self.camera_calibrate.birds_eye(edges)

        (l_x, l_y) = self.scan_frame_with_windows(flat_edges, self.l_windows)
        self.leftLane.process_points(l_x, l_y)
        (r_x, r_y) = self.scan_frame_with_windows(flat_edges, self.r_windows)
        self.rightLane.process_points(r_x, r_y)

        # f, (ax1) = plt.subplots(1, 1, figsize=(9, 6))
        # ax1.imshow(debug_img)
        # ax1.set_title('out_img', fontsize=20)
        # plt.axis('off')
        # plt.show()

        debug_overlay = self.draw_debug_overlay(flat_edges, lines=True, windows=True)
        debug_overlay = cv2.resize(debug_overlay, (0, 0), fx=0.3, fy=0.3)
        (h, w, _) = debug_overlay.shape
        output_img[20:20 + h, 20:20 + w, :] = debug_overlay

        text_x = 20 + 20 + w + w + 20

        curvature = int(np.average([self.leftLane.radius_of_curvature(), self.rightLane.radius_of_curvature()]))
        text = 'Radius of curvature:  {} m'.format(curvature)
        cv2.putText(output_img, text, (text_x, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

        center_dis = (3.7/2) - self.leftLane.camera_distance()
        direction = 'left'
        if center_dis < 0:
            direction = 'right'
        text = '{:.2f} m  {} of center'.format(center_dis, direction)

        cv2.putText(output_img, text, (text_x, 110), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)


        output_img = self.draw_lane_overlay(output_img, unwarp_matrix)
        return output_img


    def scan_frame_with_windows(self, img, windows):
        """
        Scans a frame using initialised windows in an attempt to track the lane edges.
        Parameters
        ----------
        img   : New frame
        windows : Array of windows to use for scanning the frame.
        Returns
        -------
        A tuple of arrays containing coordinates of points found in the specified windows.
        """
        indices = np.empty([0], dtype=np.int)
        nonzero = img.nonzero()
        window_x = None
        for window in windows:
            indices = np.append(indices, window.pixels_in(nonzero, window_x), axis=0)
            window_x = window.mean_x
        return (nonzero[1][indices], nonzero[0][indices])



    def draw_debug_overlay(self, binary_img):

        image = np.dstack((binary_img, binary_img, binary_img))


        for window in self.l_windows:
            coordinates = window.coordinates()
            cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)
        for window in self.r_windows:
            coordinates = window.coordinates()
            cv2.rectangle(image, coordinates[0], coordinates[1], (1., 1., 0), 2)

        cv2.polylines(image, [self.leftLane.get_points()], False, (1., 0, 0), 2)
        cv2.polylines(image, [self.rightLane.get_points()], False, (1., 0, 0), 2)
        return image * 255


    def draw_lane_overlay(self, image, unwarp_matrix=None):
        """
        Draws an overlay with tracked lane applying perspective unwarp to project it on the original frame.
        Parameters
        ----------
        image           : Original frame.
        unwarp_matrix   : Transformation matrix to unwarp the bird's eye view to initial frame. Defaults to `None` (in
        which case no unwarping is applied).
        Returns
        -------
        Frame with a lane overlay.
        """
        # Create an image to draw the lines on
        overlay = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.leftLane.get_points(), np.flipud(self.rightLane.get_points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))
        cv2.polylines(overlay, np.int32([self.leftLane.get_points()]), isClosed=False, color=(255, 0, 0), thickness=25)
        cv2.polylines(overlay, np.int32([self.rightLane.get_points()]), isClosed=False, color=(0, 0, 255), thickness=25)

        if unwarp_matrix is not None:
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            overlay = cv2.warpPerspective(overlay, unwarp_matrix, (image.shape[1], image.shape[0]))
        # Combine the result with the original image
        return cv2.addWeighted(image, 1, overlay, 0.3, 0)

    def gradient_abs_value_mask(self, image, sobel_kernel=3, axis='x', threshold=(0, 255)):
        """
        Masks the image based on gradient absolute value.

        Parameters
        ----------
        image           : Image to mask.
        sobel_kernel    : Kernel of the Sobel gradient operation.
        axis            : Axis of the gradient, 'x' or 'y'.
        threshold       : Value threshold for it to make it to appear in the mask.

        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Take the absolute value of derivative in x or y given orient = 'x' or 'y'
        if axis == 'x':
            sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if axis == 'y':
            sobel = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Scale to 8-bit (0 - 255) then convert to type = np.uint8
        sobel = np.uint8(255 * sobel / np.max(sobel))
        # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
        mask = np.zeros_like(sobel)
        # Return this mask as your binary_output image
        mask[(sobel >= threshold[0]) & (sobel <= threshold[1])] = 1
        return mask

    def gradient_magnitude_mask(self, image, sobel_kernel=3, threshold=(0, 255)):
        """
        Masks the image based on gradient magnitude.

        Parameters
        ----------
        image           : Image to mask.
        sobel_kernel    : Kernel of the Sobel gradient operation.
        threshold       : Magnitude threshold for it to make it to appear in the mask.

        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Take the gradient in x and y separately
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the magnitude
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # Scale to 8-bit (0 - 255) and convert to type = np.uint8
        magnitude = (magnitude * 255 / np.max(magnitude)).astype(np.uint8)
        # Create a binary mask where mag thresholds are met
        mask = np.zeros_like(magnitude)
        mask[(magnitude >= threshold[0]) & (magnitude <= threshold[1])] = 1
        # Return this mask as your binary_output image
        return mask

    def gradient_direction_mask(self, image, sobel_kernel=3, threshold=(0, np.pi / 2)):
        """
        Masks the image based on gradient direction.

        Parameters
        ----------
        image           : Image to mask.
        sobel_kernel    : Kernel of the Sobel gradient operation.
        threshold       : Direction threshold for it to make it to appear in the mask.

        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        # Take the gradient in x and y separately
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the x and y gradients and calculate the direction of the gradient
        direction = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))
        # Create a binary mask where direction thresholds are met
        mask = np.zeros_like(direction)
        # Return this mask as your binary_output image
        mask[(direction >= threshold[0]) & (direction <= threshold[1])] = 1
        return mask

    def color_threshold_mask(self, image, threshold=(0, 255)):
        """
        Masks the image based on color intensity.

        Parameters
        ----------
        image           : Image to mask.
        threshold       : Color intensity threshold.

        Returns
        -------
        Image mask with 1s in activations and 0 in other pixels.
        """
        mask = np.zeros_like(image)
        mask[(image > threshold[0]) & (image <= threshold[1])] = 1
        return mask

    def get_edges(self, image):

        # Convert to HLS color space and separate required channel
        hls = cv2.cvtColor(np.copy(image), cv2.COLOR_RGB2HLS).astype(np.float)
        s_channel = hls[:, :, 2]
        # Get a combination of all gradient thresholding masks
        gradient_x = self.gradient_abs_value_mask(s_channel, axis='x', sobel_kernel=3, threshold=(20, 100))
        gradient_y = self.gradient_abs_value_mask(s_channel, axis='y', sobel_kernel=3, threshold=(20, 100))
        magnitude = self.gradient_magnitude_mask(s_channel, sobel_kernel=3, threshold=(20, 100))
        direction = self.gradient_direction_mask(s_channel, sobel_kernel=3, threshold=(0.7, 1.3))
        gradient_mask = np.zeros_like(s_channel)
        gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
        # Get a color thresholding mask
        color_mask = self.color_threshold_mask(s_channel, threshold=(170, 255))

        # if separate_channels:
        #     return np.dstack((np.zeros_like(s_channel), gradient_mask, color_mask))
        # else:
        #     mask = np.zeros_like(gradient_mask)
        #     mask[(gradient_mask == 1) | (color_mask == 1)] = 1
        #     return mask

        mask = np.zeros_like(gradient_mask)
        mask[(gradient_mask == 1) | (color_mask == 1)] = 1
        return mask



