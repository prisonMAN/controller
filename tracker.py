#!/usr/bin/env python
# coding: UTF-8
import apriltag
import cv2
import time
import numpy as np

from kalman import KalmanFilter

# （1）类初始化__init__
# （2）初始化状态（mean)与状态协方差(covariance)的函数initiate
# （3）预测阶段函数predict 
# （4）分布转换函数project
# （5) 更新阶段函数update
# （6) 计算状态分布和测量（检测框）之间距离函数gating_distance


class TargetTracker:
    def __init__(self):
        self.MaxTrackingDistance = 0.8
        self.KFStateReset()
        self.m_PrevTime = time.perf_counter()


    def KFStateReset(initialPosVec):
        f = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1]])

        # H - measurement matrix
        h = np.eye(3)

        # Q - process noise covariance matrix
        q = np.diag([1, 1, 1, 1, 1, 1])

        # R - measurement noise covariance matrix
        rconst = 0.03
        r = np.diag([rconst, rconst, rconst])

        # P - error estimate covariance matrix
        p = np.eye(6)

        # Xpost
        Xpost = np.concatenate((initialPosVec, [0, 0, 0]))

        Kf = KalmanFilter({'F': f, 'H': h, 'Q': q, 'R': r, 'P': p, 'Xpost': Xpost})

        return Kf
        
    def trackTarget(self, x, y, z):
        currentTime = time.perf_counter()
        timeDiff = currentTime - m_PrevTime

        temp = self.Kf.predict(timeDiff)
        distance = np.linalg.norm(temp - [x, y, z])

        if(distance < self.MaxTrackingDistance): # 距离小于阈值，认为是同一个目标，后续可以加别的条件
            targetstate = self.update(x, y, z)
        else:
            targetstate = None # 各种原因引起的误识别，串口发送上一次发送的值

        m_PrevTime = currentTime

        return targetstate
        