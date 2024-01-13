#!/usr/bin/env python
# coding: UTF-8
import cv2 as cv
import numpy as np
import pandas as pd

from controller import Controller
from detection import TargetDetector


cap = cv.VideoCapture(0)
detector = TargetDetector()
controller = Controller()


def exports(arr, filename):
    df = pd.DataFrame(arr.reshape(1, -1), columns=["x", "y", "z", "yaw", "pitch", "roll"])
    df.to_csv(filename, mode='a', index=False, header=False)


while 1:
    # 获得图像
    ret, frame = cap.read()
    # 检测按键
    k = cv.waitKey(1)
    if k == 27:
        break

    arr = detector.run(frame)
    controller.run(arr[0], arr[1], arr[2])


cap.release()
cv.destroyAllWindows()