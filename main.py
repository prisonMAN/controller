#!/usr/bin/env python
# coding: UTF-8
import apriltag
import cv2
import time
import numpy as np

from kalman import KalmanFilter
from tracker import TargetTracker
from detection import TargetDetector

cap = cv2.VideoCapture(0)

while (1):
    # 获得图像
    ret, frame = cap.read()
    # 检测按键
    k = cv2.waitKey(1)
    if k == 27:
        break

    tags = TargetDetector.getTarget(frame)
    status = TargetDetector.getPose(tags)
    status[0], status[1], status[2] = TargetTracker.trackTarget(status[0], status[1], status[2])

cap.release()
cv2.destroyAllWindows()