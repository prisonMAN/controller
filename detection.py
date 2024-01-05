import numpy as np
import cv2 as cv
import apriltag
import math


class PoseEstimator():
    def __init__(self):
        self.at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0
        self.pitch = 0
        self.roll = 0

        self.camera_matrix = np.array(([598.29493, 0, 304.76898],
                                       [0, 597.56086, 233.34762],
                                       [0, 0, 1.0]), dtype=np.double)
        self.dist_coefs = np.array([-0.53572, 1.35993, -0.00244, 0.00620, 0.00000], dtype=np.double)

        self.object_3d_points = np.array(([-80, -80, 0],
                                          [80, -80, 0],
                                          [80, 80, 0],
                                          [-80, 80, 0]), dtype=np.double)

        self.display_origin = True  # 展示源图像
        self.display_debug = True

    def getPose(self, src):
        if self.display_origin:
            cv.imshow("Origin", src)

        # 检测apriltag
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)

        if len(tags) > 0:
            left_top = tags[0].corners[0].astype(int)
            right_top = tags[0].corners[1].astype(int)
            right_bottom = tags[0].corners[2].astype(int)
            left_bottom = tags[0].corners[3].astype(int)

            object_2d_point = np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.double)

            # 求解相机位姿
            found, rvec, tvec = cv.solvePnP(self.object_3d_points, object_2d_point, self.camera_matrix, self.dist_coefs)
            rotM = cv.Rodrigues(rvec)[0]
            camera_postion = -np.matrix(rotM).T * np.matrix(tvec)

            self.x = tvec[0]
            self.y = tvec[1]
            self.z = tvec[2]

            self.yaw = math.atan2(rotM[2][1], rotM[2][2]) * (180 / math.pi)
            self.pitch = math.atan2(-rotM[2][0], math.sqrt(rotM[2][1] * rotM[2][1] + rotM[2][2]) * rotM[2][2]) * (
                    180 / math.pi)
            self.roll = math.atan2(rotM[1][0], rotM[0][0]) * (180 / math.pi)

    def show(self, src):
        pass

