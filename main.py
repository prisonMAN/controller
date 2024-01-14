#!/usr/bin/env python
# coding: UTF-8
import cv2 as cv
import numpy as np
import pandas as pd
import gxipy as gx

from controller import Controller
from detection import TargetDetector


detector = TargetDetector()
controller = Controller()


def exports(arr, filename):
    df = pd.DataFrame(arr.reshape(1, -1), columns=["x", "y", "z", "yaw", "pitch", "roll"])
    df.to_csv(filename, mode='a', index=False, header=False)


# create a device manager
device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
if dev_num == 0:
    print("Number of enumerated devices is 0")

# open the first device
cam = device_manager.open_device_by_index(1)

# exit when the camera is a mono camera
if cam.PixelColorFilter.is_implemented() is False:
    print("This sample does not support mono camera.")
    cam.close_device()

# set continuous acquisition
cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

# set exposure
cam.ExposureTime.set(5000.0)
cam.ExposureAuto.set(0)

# set gain
cam.GainAuto.set(0)
cam.Gain.set(2.0)

cam.BalanceWhiteAuto.set(0)

# get param of improving image quality
if cam.GammaParam.is_readable():
    gamma_value = cam.GammaParam.get()
    gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
else:
    gamma_lut = None
if cam.ContrastParam.is_readable():
    contrast_value = cam.ContrastParam.get()
    contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
else:
    contrast_lut = None
if cam.ColorCorrectionParam.is_readable():
    color_correction_param = cam.ColorCorrectionParam.get()
else:
    color_correction_param = 0

# start data acquisition
cam.stream_on()

while 1:
    # 获得图像
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("Getting image failed.")
        continue

    # get RGB image from raw image
    rgb_image = raw_image.convert("RGB")
    if rgb_image is None:
        continue

    # improve image quality
    rgb_image.image_improvement(color_correction_param, contrast_lut, gamma_lut)

    # create numpy array with data from raw image
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None:
        continue

    frame = cv.cvtColor(numpy_image, cv.COLOR_BGR2RGB)
    # 检测按键
    k = cv.waitKey(1)
    if k == 27:
        break

    arr = detector.run(frame)
    controller.run(arr[0], arr[1], arr[2])

# stop data acquisition
cam.stream_off()

# close device
cam.close_device()
cv.destroyAllWindows()