import numpy as np
import cv2
import pickle

class camera(object):

    def __init__(self):

        self.camera_matrix = None
        self.dist_coef = None
        self.calibration_images_success = []
        self.calibration_images_fail = []

    def load_calibration_data(self, datafilename='camera_calibration_data.p'):
        calibration_data = pickle.load(open(datafilename, "rb"))
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coef = calibration_data['distortion_coefficient']

    def undistort(self, img):
        # calibration_data = pickle.load(open("camera_calibration_data.p", "rb"))
        # camera_matrix = calibration_data['camera_matrix']
        # dist_coef = calibration_data['distortion_coefficient']
        if self.camera_matrix is not None and self.dist_coef is not None:
            undist = cv2.undistort(img, self.camera_matrix, self.dist_coef, None, self.camera_matrix)
            return undist
        else:
            print('check camera_matrix and dist_coef')
            return img


    # Perform perspective transform
    def birds_eye(self, img):
        undist = self.undistort(img)
        # img_size = (undist.shape[1], undist.shape[0])
        # src = np.float32([(580, 460),  # top left
        #                   (200, 720),  # bottom left
        #                   (1110, 720),  # bottom right
        #                   (703, 460)])  # top right
        #
        # dst = np.float32([(320, 0),  # top left
        #                   (320, 720),  # bottom left
        #                   (960, 720),
        #                   (960, 0)])

        # Get image dimensions
        (h, w) = (undist.shape[0], undist.shape[1])

        # # Define source points
        src = np.float32([[w // 2 - 76, h * .625], [w // 2 + 76, h * .625], [-100, h], [w + 100, h]])
        # print(src)
        # Define corresponding destination points
        dst = np.float32([[100, 0], [w - 100, 0], [100, h], [w - 100, h]])
        # print(dst)

        transform_matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, transform_matrix, (w, h))
        unwarp_matrix = cv2.getPerspectiveTransform(dst, src)
        return warped, unwarp_matrix

    def apply_thresholds(self, img):

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
        return combined_binary

    def calibration(self, images=[], x_cor=9, y_cor=6, outputfilename='camera_calibration_data.p'):

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((y_cor * x_cor, 3), np.float32)
        objp[:, :2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to grayscale
            ret, corners = cv2.findChessboardCorners(gray, (x_cor, y_cor), None)  # Find the chessboard corners
            # If found, add object points, image points
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)
                cv2.drawChessboardCorners(img, (x_cor, y_cor), corners, ret)
                self.calibration_images_success.append(img)
            else:
                self.calibration_images_fail.append(fname)

        print('Corners were found on', str(len(imgpoints)), 'out of', str(len(images)), 'it is', str(
            len(imgpoints) * 100.0 / len(images)), '% of calibration images')
        img1 = cv2.imread(images[0])
        h, w = img1.shape[:2]
        ret, camera_matrix, distortion_coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None,
                                                                                None)
        calibration_data = {
            "camera_matrix": camera_matrix,
            "distortion_coefficient": distortion_coef
        }
        self.camera_matrix = calibration_data['camera_matrix']
        self.dist_coef = calibration_data['distortion_coefficient']
        pickle.dump(calibration_data, open(outputfilename, "wb"))