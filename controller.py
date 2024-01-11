import numpy as np


class Controller:
    def __init__(self, x, y, z):
        self.x0 = x
        self.y0 = y
        self.z0 = z
        self.width = 400
        self.length = 300
        self.height = 400
        self.stride = 2
        self.deadzone = 1

    def calDisDiff(self, x, y, z):
        return np.array([x - self.x0, y - self.y0, z - self.z0])

    def restrictArea(self, x, y, z):
        if x > self.length or y > self.width or z > self.height:
            print("Out of range!!!")

    def

