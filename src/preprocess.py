import os
import glob
import sys, argparse
import pprint
import numpy as np
import cv2

DATA_DIR = os.path.join('images', 'calibrated') + os.sep  ## input calibrated images
save_dir = os.path.join('images', 'save_pre') + os.sep  ## unified output for intrinsic.py
os.makedirs(save_dir, exist_ok=True)
images = [each for each in glob.glob(DATA_DIR + "*.png")]
images = sorted(images)
for each in images:
    grayImage = cv2.imread(each, 0)
    dst = cv2.bilateralFilter(grayImage, 0, 50, 30)  # 双边滤波
    ret, thresh =cv2.threshold(dst, 45, 255,cv2.THRESH_BINARY)
    # cv2.imwrite(each[:-4]+'1'+'.png', thresh)
    # yield (each, thresh)
    medianblur = cv2.medianBlur(np.uint8(thresh), 5)
    cv2.imwrite(os.path.join(save_dir, os.path.basename(each)[-8:]), medianblur)