
from source.camera import camera
# import source.camera

# from .Camera import CameraCalibration

import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import cv2



if __name__ == "__main__":
    print('main')

    images = glob.glob('../camera_cal/calibration*.jpg')
    # print(images)

    camera_calibrate = camera()
    # camera_calibrate.calibration(images, x_cor=9, y_cor=6, outputfilename='./camera_calibration_data.p')
    #
    # # images = sorted(images, key=lambda x: float(re.findall("(\d+)", x)[0]))
    #
    # print('Correction images (successfully detected corners):')
    # plt.figure(figsize=(11.5, 9))
    # gridspec.GridSpec(6, 3)
    # # Step through the list and search for chessboard corners
    # for i, image in enumerate(camera_calibrate.calibration_images_success):
    #     plt.subplot2grid((6, 3), (i // 3, i % 3), colspan=1, rowspan=1)
    #     plt.imshow(image)
    #     plt.axis('off')
    # plt.show()
    #
    # plt.figure(figsize=(12, 4))
    # plt.figtext(.5, .8, 'Images in which cv2 failed to find desired corners', fontsize=22, ha='center')
    # for i, p in enumerate(camera_calibrate.calibration_images_fail):
    #     plt.subplot(1, 3, i + 1)
    #     plt.imshow(mpimg.imread(p))  # draw the first image of each class
    #     plt.title(p)
    #     plt.axis('off')
    # plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    # plt.show()
    # plt.savefig('fail.png')

    camera_calibrate.load_calibration_data('./camera_calibration_data.p')

    # orig_img = mpimg.imread('../test_images/test1.jpg')
    # undist_img = camera_calibrate.undistort(orig_img)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    # ax1.imshow(orig_img)
    # ax1.set_title('Original', fontsize=20)
    # ax2.imshow(undist_img)
    # ax2.set_title('Undistorted', fontsize=20)
    # # plt.show()
    # plt.savefig('undistort2.png')

    # i = 1
    # for image in glob.glob('../test_images/*.jpg'):
    #     orig_img = cv2.imread(image)
    #     birdeye_img = camera_calibrate.birds_eye(orig_img)
    #     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    #     f.tight_layout()
    #     ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    #     ax1.set_title('Original', fontsize=20)
    #     ax2.imshow(cv2.cvtColor(birdeye_img, cv2.COLOR_BGR2RGB))
    #     ax2.set_title('Undistorted and Warped Image', fontsize=20)
    #     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #     # plt.show()
    #     plt.savefig('../output_images/warp_' + str(i) + '.png')
    #     i += 1


