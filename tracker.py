#!/usr/bin/env python
# coding: UTF-8
import time
import numpy as np
import cv2 as cv

from kalman import KalmanFilter


class TargetTracker:
    def __init__(self):
        self.Kf = None
        self.MaxTrackingDistance = 122222
        self.KFStateReset()
        self.m_PrevTime = time.perf_counter()

    def KFStateReset(self):
        f = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        # H - measurement matrix
        h = np.eye(3, 6)

        # Q - process noise covariance matrix
        q = np.diag([1, 1, 1, 1, 1, 1])

        # R - measurement noise covariance matrix
        rconst = 0.03
        r = np.diag([rconst, rconst, rconst])

        # P - error estimate covariance matrix
        p = np.eye(6)

        # Xpost
        Xpost = np.zeros(6)

        self.Kf = KalmanFilter({'F': f, 'H': h, 'Q': q, 'R': r, 'P': p, 'Xpost': Xpost})

    def trackTarget(self, x, y, z):
        currentTime = time.perf_counter()
        timeDiff = currentTime - self.m_PrevTime

        temp = self.Kf.predict(timeDiff)
        distance = np.linalg.norm(temp[:3] - [x, y, z])

        if distance < self.MaxTrackingDistance:  # 距离小于阈值，认为是同一个目标，后续可以加别的条件
            targetstate = self.Kf.update(np.array([x, y, z]))
            print("1")
        else:
            self.KFStateReset()
            targetstate = None  # 各种原因引起的误识别，串口发送上一次发送的值

        self.m_PrevTime = currentTime

        print(targetstate)

        return targetstate
