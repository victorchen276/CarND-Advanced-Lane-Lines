
from source.camera import camera
from source.LaneDetect import LaneDetect

from moviepy.editor import VideoFileClip
import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import cv2

#
# def process_video(input_video_file):
#     clip1 = VideoFileClip(input_video_file);
#     outputclip = clip1.fl_image(process_vid)
#     outputclip.write_videofile('output_'+input_video_file, audio=False);

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])



if __name__ == "__main__":
    print('main')

    # images = glob.glob('../camera_cal/calibration*.jpg')
    # print(images)

    camera = camera()
    # camera.calibration(images, x_cor=9, y_cor=6, outputfilename='./camera_calibration_data_1')
    camera.load_calibration_data('./camera_calibration_data.p')


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

    # camera_calibrate.load_calibration_data('./camera_calibration_data.p')

    # orig_img = mpimg.imread('../test_images/test1.jpg')
    # undist_img = camera_calibrate.undistort(orig_img)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    # ax1.imshow(orig_img)
    # ax1.set_title('Original', fontsize=20)
    # ax2.imshow(undist_img)
    # ax2.set_title('Undistorted', fontsize=20)
    # # plt.show()
    # plt.savefig('undistort2.png')

    # Perspective transform
    # for image in glob.glob('../test_images/*.jpg'):
    #     orig_img = cv2.imread(image)
    #     birdeye_img, _ = camera.birds_eye(orig_img)
    #     f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
    #     f.tight_layout()
    #     ax1.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    #     ax1.set_title('Original', fontsize=20)
    #     ax2.imshow(cv2.cvtColor(birdeye_img, cv2.COLOR_BGR2RGB))
    #     ax2.set_title('Undistorted and Warped Image', fontsize=20)
    #     plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #     plt.show()
    #     # plt.savefig('../output_images/warp_' + str(i) + '.png')



    # # edege
    # image = mpimg.imread('../test_images/test6.jpg')
    # lane_detecter = LaneDetect()
    # result = lane_detecter.get_edges(image)
    #
    # # Plot the result
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    # # f.tight_layout()
    # ax1.axis('off')
    # ax1.imshow(image)
    # ax1.set_title('Original', fontsize=18)
    # ax2.axis('off')
    # ax2.set_title('Edge', fontsize=18)
    #
    #
    # ax2.imshow(result, cmap='gray')
    # plt.show()
    # plt.savefig('edge.png')

    # # Detect Lane line
    # for image_name in glob.glob('../test_images/*.jpg'):
    #     orig_img = mpimg.imread(image_name)
    #
    #     lane_detecter = LaneDetect()
    #     lane_detecter.initcamera()
    #     lane_detecter.initlines(orig_img)
    #     output_img = lane_detecter.process_pipeline(orig_img)
    #     f, (ax1) = plt.subplots(1, 1, figsize=(9, 6))
    #     ax1.imshow(output_img)
    #     ax1.set_title('output_img', fontsize=20)
    #     plt.axis('off')
    #     plt.show()
    #     break

    # Applying pipeline to video
    clip1 = VideoFileClip('../project_video.mp4')
    lane_detecter = LaneDetect()
    lane_detecter.initcamera()
    lane_detecter.initlines(clip1.get_frame(0))
    outputclip = clip1.fl_image(lane_detecter.process_pipeline)
    outputclip.write_videofile('../output_videos/output_project_video.mp4', audio=False)
    #
    # clip1 = VideoFileClip('../harder_challenge_video.mp4');
    # lane_detecter = LaneDetect(clip1.get_frame(0))
    # outputclip = clip1.fl_image(lane_detecter.process_pipeline)
    # outputclip.write_videofile('../output_harder_challenge_video.mp4', audio=False)
    #
    # clip1 = VideoFileClip('../challenge_video.mp4')
    # lane_detecter = LaneDetect(clip1.get_frame(0))
    # outputclip = clip1.fl_image(lane_detecter.process_pipeline)
    # outputclip.write_videofile('../output_challenge_video.mp4', audio=False)