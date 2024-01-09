from typing import Any

import numpy as np
import cv2 as cv
import apriltag
import math
import sys

from numpy import ndarray, dtype


class TargetDetector:
    def __init__(self):
        self.at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
        self.x = None
        self.y = None
        self.z = None
        self.yaw = None
        self.pitch = None
        self.roll = None
        self.arr_post = arr = np.empty((6,))

        self.camera_matrix = np.array(([598.29493, 0, 304.76898],
                                       [0, 597.56086, 233.34762],
                                       [0, 0, 1.0]), dtype=np.double)
        self.dist_coefs = np.array([-0.53572, 1.35993, -0.00244, 0.00620, 0.00000], dtype=np.double)

        self.object_2d_point = None
        self.object_3d_points = np.array(([-80, -80, 0],
                                          [80, -80, 0],
                                          [80, 80, 0],
                                          [-80, 80, 0]), dtype=np.double)

        self.display_origin = True  # 展示源图像
        self.display_debug = True

    def getTarget(self, src):
        if self.display_origin:
            cv.imshow("Origin", src)

        # 检测apriltag
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        tags = self.at_detector.detect(gray)

        return tags

    def getPose(self, src, tags):
        if len(tags) > 0:
            left_top = tags[0].corners[0].astype(int)
            right_top = tags[0].corners[1].astype(int)
            right_bottom = tags[0].corners[2].astype(int)
            left_bottom = tags[0].corners[3].astype(int)

            self.object_2d_point = np.array([left_top, right_top, right_bottom, left_bottom], dtype=np.double)

            # 求解相机位姿
            found, rvec, tvec = cv.solvePnP(self.object_3d_points, self.object_2d_point, self.camera_matrix,
                                            self.dist_coefs)
            rotM = cv.Rodrigues(rvec)[0]

            self.x = tvec[0]
            self.y = tvec[1]
            self.z = tvec[2]

            self.yaw = math.atan2(rotM[2][1], rotM[2][2]) * (180 / math.pi)
            self.pitch = math.atan2(-rotM[2][0], math.sqrt(rotM[2][1] * rotM[2][1] + rotM[2][2]) * rotM[2][2]) * (
                    180 / math.pi)
            self.roll = math.atan2(rotM[1][0], rotM[0][0]) * (180 / math.pi)

            arr = np.empty((6,))

            # Assign values to the array elements
            arr[0] = self.x
            arr[1] = self.y
            arr[2] = self.z
            arr[3] = self.yaw
            arr[4] = self.pitch
            arr[5] = self.roll

            self.arr_post = arr

            if self.display_debug is True:
                if self.object_2d_point is not None:
                    # 将 self.object_2d_point[0] 转换为整数，并将其作为元组的一部分传递给 cv.circle 函数
                    cv.circle(src, tuple(self.object_2d_point[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
                    cv.circle(src, tuple(self.object_2d_point[1].astype(int)), 4, (255, 0, 0), 2)  # right-top
                    cv.circle(src, tuple(self.object_2d_point[2].astype(int)), 4, (255, 0, 0), 2)  # right-bottom
                    cv.circle(src, tuple(self.object_2d_point[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom

                    text1 = f'x: {arr.astype(int)[0]}, y: {arr.astype(int)[1]}'
                    text2 = f'z: {arr.astype(int)[2]}'
                    text3 = f'yaw: {arr.astype(int)[3]}, pitch: {arr.astype(int)[4]}'
                    text4 = f'roll: {arr.astype(int)[5]}'
                    cv.putText(src, text1, (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                    cv.putText(src, text2, (10, 40), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                    cv.putText(src, text3, (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                    cv.putText(src, text4, (10, 80), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
                else:
                    cv.putText(src, '未识别到，发送滤波数据', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)

                cv.imshow('Debug', src)
            print(arr.astype(int))

            return arr
        else:
            return self.arr_post  # 如果全为0则返回上一次观测的数据

    def run(self, src) -> ndarray[Any, dtype[Any]]:
        return self.getPose(src, self.getTarget(src))
