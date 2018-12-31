import numpy as np
import cv2

from source.camera import camera

class LaneDetect(object):

    def __init__(self):
        self.leftLane = None
        self.rightLane = None

        self.camera_calibrate = camera()
        self.camera_calibrate.load_calibration_data('./camera_calibration_data.p')

    def process_pipeline(self, img):
        print('process_pipeline')
