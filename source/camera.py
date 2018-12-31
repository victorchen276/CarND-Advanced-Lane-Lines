import numpy as np
import cv2
import matplotlib.image as mpimg
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
        img_size = (undist.shape[1], undist.shape[0])
        src = np.float32([(580, 460),  # top left
                          (200, 720),  # bottom left
                          (1110, 720),  # bottom right
                          (703, 460)])  # top right

        dst = np.float32([(320, 0),  # top left
                          (320, 720),  # bottom left
                          (960, 720),
                          (960, 0)])

        transform_matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(undist, transform_matrix, img_size)
        return warped

    def calibration(self, images=[], x_cor=9, y_cor=6, outputfilename='camera_calibration_data.p'):

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((y_cor * x_cor, 3), np.float32)
        objp[:, :2] = np.mgrid[0:x_cor, 0:y_cor].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.


        # Step through the list and search for chessboard corners
        # corners_not_found = []  # Calibration images in which opencv failed to find corners
        # plt.figure(figsize=(12, 18))  # Figure for calibration images
        # plt.figtext(0.5, 0.9, 'Image with corners patterns drawn', fontsize=22, ha='center')
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Conver to grayscale
            ret, corners = cv2.findChessboardCorners(gray, (x_cor, y_cor), None)  # Find the chessboard corners
            # If found, add object points, image points
            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)
                # plt.subplot(5, 4, len(imgpoints))

                cv2.drawChessboardCorners(img, (x_cor, y_cor), corners, ret)
                # plt.imshow(img)
                # plt.title(fname)
                # plt.axis('off')
                self.calibration_images_success.append(img)
            else:
                self.calibration_images_fail.append(fname)
                # corners_not_found.append(fname)
        # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        # plt.show()
        # plt.savefig('success.png')

        print('Corners were found on', str(len(imgpoints)), 'out of', str(len(images)), 'it is', str(
            len(imgpoints) * 100.0 / len(images)), '% of calibration images')
        # Draw pictures
        # plt.figure(figsize=(12, 4))
        # # plt.figtext(.5, .8, 'Images in which cv2 failed to find desired corners', fontsize=22, ha='center')
        # for i, p in enumerate(corners_not_found):
        #     plt.subplot(1, 3, i + 1)
        #     plt.imshow(mpimg.imread(p))  # draw the first image of each class
        #     plt.title(p)
        #     plt.axis('off')
        # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
        # plt.show()
        # plt.savefig('fail.png')

        img1 = cv2.imread(images[0])
        h, w = img1.shape[:2]
        ret, camera_matrix, distortion_coef, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None,
                                                                                None)

        calibration_data = {
            "camera_matrix": camera_matrix,
            "distortion_coefficient": distortion_coef
        }

        pickle.dump(calibration_data, open(outputfilename, "wb"))