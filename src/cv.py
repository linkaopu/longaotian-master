#!/usr/bin/env python
# ref:https://learnopencv.com/camera-calibration-using-opencv/
# 原文链接：https://blog.csdn.net/qq_43528254/article/details/108276225

# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # 用于数值计算
import os  # 用于文件和目录操作
import glob  # 用于文件路径匹配

# 定义棋盘格的尺寸（内角点数）
CHECKERBOARD = (5, 7)
# 设置角点检测的终止条件
# TERM_CRITERIA_EPS: 角点位置变化小于指定精度时停止
# TERM_CRITERIA_MAX_ITER: 最大迭代次数
# 30: 最大迭代次数为30次
# 0.0001: 精度为0.0001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

# 创建列表存储每张棋盘格图像的3D点和2D点
objpoints = []  # 3D点：世界坐标系中的点
imgpoints = []  # 2D点：图像坐标系中的点

# 获取当前脚本文件所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))


# 定义世界坐标系中的3D点
# 创建一个大小为(1, CHECKERBOARD[0] * CHECKERBOARD[1], 3)的零数组
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# 设置棋盘格角点的3D坐标，z坐标为0，x和y坐标根据棋盘格尺寸设置
# 每个方格的物理尺寸为1.5单位
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*1.5
# 用于存储前一张图像的尺寸
prev_img_shape = None

# 获取指定目录中的所有棋盘格图像
images = glob.glob(os.path.join('images', 'calibrated', '1', '*.png'))

# 遍历所有棋盘格图像进行角点检测
for fname in images:
    # 读取图像
    img = cv2.imread(fname)
    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    # CALIB_CB_ADAPTIVE_THRESH: 使用自适应阈值
    # CALIB_CB_FAST_CHECK: 快速检查
    # CALIB_CB_NORMALIZE_IMAGE: 图像归一化
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

    """
    如果检测到足够数量的角点，则优化角点坐标并在图像上显示
    """
    if ret == True:
        # 将3D点添加到列表中
        objpoints.append(objp)
        
        # 优化角点坐标
        # cornerSubPix: 亚像素级角点检测
        # (11, 11): 搜索窗口大小
        # (-1, -1): 死区大小
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 将优化后的2D点添加到列表中
        imgpoints.append(corners2)

        # 在图像上绘制棋盘格角点
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # 确保输出目录存在
        os.makedirs('images', exist_ok=True)
        # 保存带有角点的图像
        cv2.imwrite(os.path.join("images", fname[-8:]), img)
    else:
        # 如果未检测到足够角点，打印文件名
        print(fname)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

# h, w = img.shape[:2]

"""
执行相机标定
通过已知的3D点(objpoints)和对应的2D角点坐标(imgpoints)进行标定
"""
# 检查是否有有效的标定图像
if len(objpoints) > 0 and len(imgpoints) > 0:
    # 使用最后一张灰度图像进行标定
    # calibrateCamera: 相机标定函数
    # gray.shape[::-1]: 图像尺寸
    # None: 相机内参矩阵初始值
    # None: 畸变系数初始值
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 打印标定结果
    print("ret: \n")
    print(ret)
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
else:
    # 如果没有有效的标定图像，打印提示信息
    print("No valid checkerboard images found for calibration.")
    # 初始化默认值
    ret = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    
# 需要校正的图像目录
# datadir = os.path.join('images', 'raw_ring')
# path = os.path.join('images', 'raw_ring')
path = os.path.join('images', 'calibrated', 'backup')
# 检查目录是否存在
if not os.path.isdir(path):
    print("Raw ring directory not found:", path)
    img_list = []
else:
    # 获取目录中的所有图像文件
    img_list = os.listdir(path)

# 遍历所有需要校正的图像
for i in img_list:
    # 读取图像
    img = cv2.imread(os.path.join(path, i))
    # 获取图像尺寸
    h, w = img.shape[:2]
    
    # 计算最优相机内参矩阵和ROI
    # getOptimalNewCameraMatrix: 计算最优相机内参矩阵
    # mtx: 相机内参矩阵
    # dist: 畸变系数
    # (w, h): 图像尺寸
    # 1: 自由缩放参数
    # (w, h): 新图像尺寸
    newMatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # 校正图像
    # undistort: 图像去畸变
    # img: 输入图像
    # mtx: 相机内参矩阵
    # dist: 畸变系数
    # None: 新相机内参矩阵
    # newMatrix: 最优相机内参矩阵
    dst = cv2.undistort(img, mtx, dist, None, newMatrix)
    
    # 保存校正后的图像
    # 确保输出目录存在
    # os.makedirs(os.path.join('images', 'calibrated'), exist_ok=True)
    # 保存校正后的图像
    # cv2.imwrite(os.path.join('images', 'calibrated', i), dst)
    
    # 使用脚本目录构建输出路径
    output_path = os.path.join(script_dir, 'images', 'calibrated', '4', i)
    # 保存校正后的图像
    cv2.imwrite(output_path, dst)

# 计算重投影误差
if len(objpoints) > 0:
    # 初始化总误差
    tot_error = 0
    # 遍历所有标定图像
    for i in range(len(objpoints)):
        # 计算重投影点
        # projectPoints: 3D点投影到图像平面
        # objpoints[i]: 3D点
        # rvecs[i]: 旋转向量
        # tvecs[i]: 平移向量
        # mtx: 相机内参矩阵
        # dist: 畸变系数
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        # 计算误差
        # norm: 计算向量范数
        # NORM_L2: L2范数
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        # 累加误差
        tot_error += error

    # 输出标定参数和平均重投影误差
    print("-------------calibrated----------------")
    print('ret:\n', ret)
    print('mtx:\n', mtx)
    print('dist:\n', dist)
    print('rvecs:\n', rvecs)
    print('tvecs:\n', tvecs)
    print ("total error: ", tot_error/len(objpoints))
else:
    # 如果没有有效的标定图像，跳过误差计算
    print("No valid checkerboard images found for calibration. Skipping error calculation.")

