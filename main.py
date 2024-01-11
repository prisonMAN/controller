#!/usr/bin/env python
# coding: UTF-8
import cv2 as cv
import numpy as np
import pandas as pd

from detection import TargetDetector


cap = cv.VideoCapture(0)
detector = TargetDetector()


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

    arr = np.asarray(detector.run(frame), dtype=np.ndarray)
    exports(arr, "output.csv")


cap.release()
cv.destroyAllWindows()