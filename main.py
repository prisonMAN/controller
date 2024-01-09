#!/usr/bin/env python
# coding: UTF-8
import cv2 as cv
import numpy as np

from detection import TargetDetector
from tracker import TargetTracker

cap = cv.VideoCapture(0)
detector = TargetDetector()
tracker = TargetTracker()


while 1:
    # 获得图像
    ret, frame = cap.read()
    # 检测按键
    k = cv.waitKey(1)
    if k == 27:
        break

    arr = np.asarray(detector.run(frame), dtype=np.ndarray)
    tracker.trackTarget(arr[0], arr[1], arr[2])

cap.release()
cv.destroyAllWindows()