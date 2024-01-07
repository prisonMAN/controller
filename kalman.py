import numpy as np

class KalmanFilter:
    def __init__(self, state):
        self.F = state['F']
        self.H = state['H']
        self.Q = state['Q']
        self.R = state['R']
        self.P_post = state['P']
        self.n = self.F.shape[0]
        self.I = np.eye(self.n)
        self.X_pre = np.zeros(self.n)
        self.X_post = state['Xpost'] if state['Xpost'].shape[0] else np.zeros(self.n)

    def predict(self, T):
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = T

        self.X_pre = np.dot(self.F, self.X_post)
        self.P_pre = np.dot(np.dot(self.F, self.P_post), self.F.T) + self.Q

        self.X_post = self.X_pre
        self.P_post = self.P_pre

        return self.X_pre

    def update(self, Z):
        K = np.dot(np.dot(self.P_pre, self.H.T), np.linalg.inv(np.dot(np.dot(self.H, self.P_pre), self.H.T) + self.R))
        self.X_post = self.X_pre + np.dot(K, (Z - np.dot(self.H, self.X_pre)))
        self.P_post = np.dot((self.I - np.dot(K, self.H)), self.P_pre)

        return self.X_post