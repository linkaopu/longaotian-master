# 计算整批图像 每个像素对应多少个毫米

# 导入所需的库
import cv2  # OpenCV库，用于图像处理
import numpy as np  # 用于数值计算
import os  # 用于文件和目录操作
import glob  # 用于文件路径匹配
import math  # 用于数学计算

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

# 定义世界坐标系中的3D点
# 创建一个大小为(1, CHECKERBOARD[0] * CHECKERBOARD[1], 3)的零数组
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
# 设置棋盘格角点的3D坐标，z坐标为0，x和y坐标根据棋盘格尺寸设置
# 每个方格的物理尺寸为1.5毫米
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)*1.5
# 用于存储前一张图像的尺寸
prev_img_shape = None

# 获取指定目录中的所有棋盘格图像
# images = glob.glob('./images/uncalibrated/*.png')   #未校正的图像
images = glob.glob(os.path.join('images', 'calibrated', '1', '*.png'))   #已校正图像

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
    如果检测到足够数量的角点，则优化角点坐标
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
    else:
        # 如果未检测到足够角点，打印文件名
        print("no chessboardcorners found in image {}".format(fname))

# 创建列表存储相邻角点之间的距离
lst = []  # 存储水平方向相邻角点的距离
lst1 = []  # 存储垂直方向相邻角点的距离

# 遍历所有检测到的角点，计算相邻角点之间的距离
for i in range(len(imgpoints)):
    for j in range(len(imgpoints[i])):
        # 计算水平方向相邻角点的距离（每行最后一个角点除外）
        if j % 5 != 4:
            # 计算两个角点之间的欧几里得距离
            lst.append(np.linalg.norm(imgpoints[i][j] - imgpoints[i][j + 1]))
        
        # 计算垂直方向相邻角点的距离（最后一行角点除外）
        if j < 30:
            # 计算两个角点之间的欧几里得距离
            lst1.append(np.linalg.norm(imgpoints[i][j] - imgpoints[i][j + 5]))

# 计算平均像素距离，并转换为每个像素对应的毫米数
# 1.5: 每个棋盘格方格的实际尺寸（毫米）
print(1.5 / np.mean(np.array(lst)))  # 水平方向每个像素对应的毫米数
print(1.5 / np.mean(np.array(lst1)))  # 垂直方向每个像素对应的毫米数