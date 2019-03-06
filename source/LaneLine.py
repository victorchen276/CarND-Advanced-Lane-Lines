import numpy as np
from collections import deque


class LaneLine(object):
    def __init__(self, x, y, h, w):
        # polynomial coefficients for the most recent fit
        self.h = h
        self.w = w
        self.frame_impact = 0
        self.coefficients = deque(maxlen=5)
        self.process_points(x, y)

    def process_points(self, x, y):
        enough_points = len(y) > 0 and np.max(y) - np.min(y) > self.h * .625
        if enough_points or len(self.coefficients) == 0:
            self.fit(x, y)

    def get_points(self):
        y = np.linspace(0, self.h - 1, self.h)
        current_fit = self.averaged_fit()
        return np.stack((
            current_fit[0] * y ** 2 + current_fit[1] * y + current_fit[2],
            y
        )).astype(np.int).T

    def averaged_fit(self):
        return np.array(self.coefficients).mean(axis=0)

    def fit(self, x, y):
        self.coefficients.append(np.polyfit(y, x, 2))

    def radius_of_curvature(self):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 27 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        points = self.get_points()
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        return int(((1 + (2 * fit_cr[0] * 720 * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def camera_distance(self):
        points = self.get_points()
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.w // 2 - x) * xm_per_pix)
