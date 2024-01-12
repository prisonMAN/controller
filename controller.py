import numpy as np


class Controller:
    def __init__(self):
        self.x0 = 0
        self.y0 = 0
        self.z0 = 0
        self.xPost = None
        self.yPost = None
        self.zPost = None
        self.left = 400
        self.right = 300
        self.front = 200
        self.back = 200
        self.up = 200
        self.down = 200
        self.stride = 2
        self.deadzone = 10

    def restrictArea(self, x, y, z):
        if x - self.x0 > self.length or y - self.y0 > self.width or z - self.z0 > self.height:
            print("Out of range!!!")

    def calDisDiff(self, x, y, z):
        return np.array([x - self.xPost, y - self.yPost, z - self.zPost])

    def deadzone(self, x, y, z):
        if x - self.xPost < self.deadzone:
            x = 0
        if y - self.yPost < self.deadzone:
            y = 0
        if z - self.zPost < self.deadzone:
            z = 0
        return np.array([x, y, z] * self.stride)

    def run(self, x, y, z):
        arr = self.calDisDiff(x, y, z)
        self.restrictArea(arr[0], arr[1], arr[2])

