#!/usr/bin/env python
# coding: UTF-8
import apriltag
import cv2
import time
import numpy as np
import kalman

#kalman = kalman.KalmanFilter()
T_pre = time.time()
cap = cv2.VideoCapture(0)
at_detector = apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))

while (1):
    #时间戳
    T = time.time()
    dt = T - T_pre
    T_pre = T

    # 获得图像
    ret, frame = cap.read()
    # 检测按键
    k = cv2.waitKey(1)
    if k == 27:
        break

    # 检测apriltag
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tags = at_detector.detect(gray)

    if len(tags) > 0:
        center = tags[0].center
        #print(type(tags[0].corners[0].astype(int)))
        #print(type(np.array([tags[0].corners[0].astype(int), tags[0].corners[0].astype(int), tags[0].corners[0].astype(int), tags[0].corners[0].astype(int)], dtype=np.double)))


    for tag in tags:
        cv2.circle(frame, tuple(tag.corners[0].astype(int)), 4, (255, 0, 0), 2)  # left-top
        cv2.circle(frame, tuple(tag.corners[1].astype(int)), 4, (255, 0, 0), 2)  # right-top
        cv2.circle(frame, tuple(tag.corners[2].astype(int)), 4, (255, 0, 0), 2)  # right-bottom
        cv2.circle(frame, tuple(tag.corners[3].astype(int)), 4, (255, 0, 0), 2)  # left-bottom
    # 显示检测结果
    cv2.imshow('capture', frame)

cap.release()
cv2.destroyAllWindows()
