"""
O型圈缺陷检测系统

该系统通过图像处理技术自动检测O型圈的缺陷，包括阈值分割、形态学操作、
连通分量标记、面积计算、边界框绘制和缺陷定位等功能。
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import os

# 设置图像文件夹路径
# 注意：在Windows系统中应使用双反斜杠或原始字符串
# directory = "E:\\O-Ring Pictures\\New"  # 原始路径
# directory = "E:/O-Ring Pictures/New"  # 修改后的路径
# directory = r"E:\O-Ring Pictures\New"  # 原始路径（原始字符串）
# directory = "E:/O-Ring Pictures/New"  # 修改后的路径（正斜杠）
# directory = "E:/O-Ring Pictures/New"  # 使用正斜杠的路径


# The folder that contains the ORing images
directory = os.path.join('images', 'calibrated', '2')


# Imports and returns a list of all the files from the specified folder
def read_images(directory):
    """
    读取指定文件夹中的所有图像文件
    
    参数:
        directory (str): 图像文件夹路径
    
    返回:
        list: 包含文件路径和图像数据的列表
    """
    image_list = []
    for file in os.listdir(directory):
        image_list.append([os.path.join(directory + '/', file), cv.imread(os.path.join(directory + '/', file), 0)])
    return image_list


# Stores histogram of pixel levels from retrieved image
def image_hist(image):
    """
    计算图像的灰度直方图
    
    参数:
        image (numpy.ndarray): 输入图像
    
    返回:
        numpy.ndarray: 图像的灰度直方图（256个灰度级）
    """
    hist = np.zeros(256)
    for i in range(0, image.shape[0]):  # 遍历行
        for j in range(0, image.shape[1]):  # 遍历列
            hist[image[i, j]] += 1  # 增加对应灰度级的像素计数
    return hist


# Returns a list of the two peak values from retrieved histogram
def hist_peaks(hist):
    """
    从直方图中找到峰值点
    
    参数:
        hist (numpy.ndarray): 图像的灰度直方图
    
    返回:
        list: 峰值点的索引值
    """
    # peaks = [np.where(hist == max(hist))[0][0]]
    # peak1 = peaks[0] - 100
    # peak2 = peaks[0] + 100
    # temp_array = [hist[i] for i in range(len(hist)) if i < peak1 or i > peak2]
    # peaks.append(temp_array.index(max(temp_array) ))
    # peaks.sort()
    peaks = [np.where(hist == max(hist))[0][0]]
    all_valley = [np.where(hist <= np.mean(hist))[0]][0]
    valley = all_valley[all_valley < peaks]
    # 找距离peaks最近的valley的索引值，即灰度值
    dist = (peaks[0] - valley)
    minInxInDist = np.where(dist == min(dist))[0]
    threshIdx = valley[minInxInDist]
    return threshIdx


# Calculates a threshold value for the retrieved image
def threshold_value(image):
    """
    计算图像的阈值
    
    参数:
        image (numpy.ndarray): 输入图像
    
    返回:
        int: 计算得到的阈值
    """
    hist = image_hist(image)  # 获取图像的直方图
    tValue = hist_peaks(hist)[0]  # 获取直方图的峰值点
    return tValue


# Applies thresholding to the retrieved image and returns a single-channel binary image
def threshold(image):
    """
    对图像进行阈值处理，返回二值图像
    
    参数:
        image (numpy.ndarray): 输入图像
    
    返回:
        numpy.ndarray: 二值化后的图像（0/255）
    """
    if image is None:
        return None
    # 如果是彩色图像，转换为灰度图
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 改进的阈值分割策略
    # 首先尝试Otsu阈值
    _, bin_img = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # 如果Otsu结果不理想，使用自适应阈值
    if np.sum(bin_img == 255) < gray.shape[0] * gray.shape[1] * 0.1:  # 如果前景像素太少
        bin_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

    # 如果前景像素太多，使用更严格的阈值
    elif np.sum(bin_img == 255) > gray.shape[0] * gray.shape[1] * 0.8:  # 如果前景像素太多
        _, bin_img = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

    return bin_img.astype(np.uint8)


# Applies erosion to the retrieved image using the morphological structure and returns the new image
def erosion(image, struct):
    """
    对图像进行腐蚀操作
    
    参数:
        image (numpy.ndarray): 输入图像
        struct (list): 形态学结构元素
    
    返回:
        numpy.ndarray: 腐蚀后的图像
    """
    eImage = image.copy()
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # 检查选定像素是否匹配前景色，并且形态学结构所需的位置在图像边界内
            pixel_is_black = (image[i, j] == 0) if image.ndim == 2 else np.all(image[i, j] == 0)
            if pixel_is_black and i - 1 >= 0 and j - 1 >= 0 and i + 1 < image.shape[0] and j + 1 < image.shape[1]:
                for y in range(-1, 2):
                    for x in range(-1, 2):
                        offset_i = i + y
                        offset_j = j + x

                        # 检查被检查的位置是否不是当前像素，并且与形态学结构中的指定位置对齐
                        neighbor_is_white = (image[offset_i, offset_j] == 255) if image.ndim == 2 else np.all(
                            image[offset_i, offset_j] == 255)
                        if [offset_i, offset_j] != [i, j] and struct[y + 1][x + 1] == 1 and neighbor_is_white:
                            eImage[i, j] = 255  # 将当前像素设置为图像的背景
    return eImage


# Applies dilation to the retrieved image using the morphological structure and returns the new image
def dilation(image, struct):
    """
    对图像进行膨胀操作
    
    参数:
        image (numpy.ndarray): 输入图像
        struct (list): 形态学结构元素
    
    返回:
        numpy.ndarray: 膨胀后的图像
    """
    dImage = image.copy()
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):

            # 检查选定像素是否匹配前景色，并且形态学结构所需的位置在图像边界内
            pixel_is_white = (image[i, j] == 255) if image.ndim == 2 else np.all(image[i, j] == 255)
            if pixel_is_white and i - 1 >= 0 and j - 1 >= 0 and i + 1 < image.shape[0] and j + 1 < image.shape[1]:
                for y in range(-1, len(struct) - 1):
                    for x in range(-1, len(struct) - 1):
                        offset_i = i + y
                        offset_j = j + x

                        # 检查被检查的位置是否不是当前像素，并且与形态学结构中的指定位置对齐
                        neighbor_is_black = (image[offset_i, offset_j] == 0) if image.ndim == 2 else np.all(
                            image[offset_i, offset_j] == 0)
                        if [offset_i, offset_j] != [i, j] and struct[y + 1][x + 1] == 1 and neighbor_is_black:
                            dImage[i, j] = 0  # 将当前像素设置为图像的背景
    return dImage


# Retrieves an image and morphological structure, then applies closing which involves dilation followed by erosion and returns a new image
def closing(image, struct):
    """
    对图像进行闭运算（先膨胀后腐蚀）
    
    参数:
        image (numpy.ndarray): 输入图像
        struct (list): 形态学结构元素
    
    返回:
        numpy.ndarray: 闭运算后的图像
    """
    cImage = dilation(image, struct)
    cImage = erosion(cImage, struct)
    return cImage


# Takes in an image and performs connected component labelling, returning the list of labels for pixels at every index position on the image
def component_label(image):
    """
    对图像进行连通组件标记
    
    参数:
        image (numpy.ndarray): 输入的二值图像
    
    返回:
        numpy.ndarray: 标记后的图像，每个连通区域有不同的标签
    """
    # 创建一个与图像大小相同的2D列表，用0填充表示未标记的属性
    label_list = np.zeros((image.shape[0], image.shape[1]))
    curlab = 1  # 声明当前标签为1
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # 安全地支持单通道和3通道输入
            pixel_is_black = (image[i, j] == 0) if image.ndim == 2 else np.all(image[i, j] == 0)
            if pixel_is_black and label_list[i][j] == 0:
                label_list[i][j] = curlab
                queue = []  # 初始化一个空列表作为队列结构
                queue.append([i, j])  # 将当前像素坐标添加到队列中

                # 当队列长度大于0时循环，因为相邻坐标仍需要处理
                while len(queue) > 0:
                    item = queue.pop(0)
                    if 0 < item[0] < image.shape[0] - 1 and 0 < item[1] < image.shape[1] - 1:
                        if np.all(image[item[0] - 1, item[1]] == 0) and label_list[item[0] - 1][item[
                            1]] == 0:  # 检查上方邻居是否为前景像素且当前未标记
                            queue.append([item[0] - 1, item[1]])  # 将其坐标添加到队列
                            label_list[item[0] - 1][item[1]] = curlab  # 用当前组件标签标记它
                        if np.all(image[item[0] + 1, item[1]] == 0) and label_list[item[0] + 1][item[
                            1]] == 0:  # 检查下方邻居是否为前景像素且当前未标记
                            queue.append([item[0] + 1, item[1]])
                            label_list[item[0] + 1][item[1]] = curlab
                        if np.all(image[item[0], item[1] - 1] == 0) and label_list[item[0]][item[
                                                                                                1] - 1] == 0:  # 检查左侧邻居是否为前景像素且当前未标记
                            queue.append([item[0], item[1] - 1])
                            label_list[item[0]][item[1] - 1] = curlab
                        if np.all(image[item[0], item[1] + 1] == 0) and label_list[item[0]][item[
                                                                                                1] + 1] == 0:  # 检查右侧邻居是否为前景像素且当前未标记
                            queue.append([item[0], item[1] + 1])
                            label_list[item[0]][item[1] + 1] = curlab
                curlab += 1
    return np.array(label_list, dtype=np.uint8)


# Retrives the list of labels, a label for calculating the area and returns the number of pixels in the area with the label
def calculate_area(label_list, curlab):
    """
    计算指定标签区域的面积（像素数量）
    
    参数:
        label_list (numpy.ndarray): 标记后的图像
        curlab (int): 要计算面积的标签
    
    返回:
        int: 指定标签区域的像素数量
    """
    area = 0
    for x in range(0, len(label_list)):
        for y in range(0, len(label_list[0])):
            if label_list[x][
                y] == curlab:  # 检查当前位置的标签是否与我们要计算的标签匹配
                area += 1
    return area


# Retrieves an image and the list of labels then returns the new image with painted labels
def paint_labels(image, label_list):
    """
    绘制标记后的图像
    
    参数:
        image (numpy.ndarray): 原始图像
        label_list (numpy.ndarray): 标记后的图像
    
    返回:
        numpy.ndarray: 绘制了标记的图像
    """
    pImage = image.copy()
    unique_labels = np.unique(label_list)
    print(f"Debug: paint_labels - Found labels: {unique_labels}")

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            # 确保乘法不会溢出，使用条件逻辑
            label_value = label_list[x][y]
            if label_value > 0:
                # 对于非零标签，设置为255（白色）以避免溢出问题
                pixel_value = 255
            else:
                pixel_value = 0
            pImage[
                x, y] = pixel_value  # 通过将标记的像素乘以255来更新，将前景色设置为白色，背景色设置为黑色

    # 统计前景像素数量
    foreground_pixels = np.sum(pImage == 255)
    total_pixels = pImage.shape[0] * pImage.shape[1]
    print(
        f"Debug: paint_labels - Foreground pixels: {foreground_pixels}/{total_pixels} ({foreground_pixels / total_pixels * 100:.2f}%)")

    return pImage


# Retrieves a list of labels and removes the smallest areas
def remove_smallest_areas(label_list):
    """
    移除最小的区域（噪声区域）
    
    参数:
        label_list (numpy.ndarray): 标记后的图像
    
    返回:
        numpy.ndarray: 移除小区域后的图像
    """
    unique_labels = np.unique(label_list)  # 提取唯一的组件标签列表
    print(f"Debug: Found {len(unique_labels)} unique labels: {unique_labels}")

    if len(unique_labels) > 2:  # 如果有多个连通区域
        unique_labels = unique_labels[1:]  # 排除背景标签0
        areas = []
        for i in range(len(unique_labels)):
            area = calculate_area(label_list, unique_labels[i])
            areas.append(area)
            print(f"Debug: Label {unique_labels[i]} has area {area}")

        # 改进的噪声区域移除策略
        # 计算所有区域的平均面积
        mean_area = np.mean(areas)
        print(f"Debug: Mean area = {mean_area}")

        # 移除面积小于平均面积20%的区域，而不是简单地移除最小区域
        threshold_area = mean_area * 0.2
        print(f"Debug: Threshold area = {threshold_area}")

        new_labels = label_list.copy()
        removed_count = 0
        for i in range(len(unique_labels)):
            if areas[i] < threshold_area:
                # 将小于阈值的区域标记为背景
                new_labels[new_labels == unique_labels[i]] = 0
                removed_count += 1
                print(f"Debug: Removed label {unique_labels[i]} with area {areas[i]}")

        print(f"Debug: Removed {removed_count} small areas")

        # 检查是否还有剩余区域
        remaining_labels = np.unique(new_labels)
        print(f"Debug: Remaining labels: {remaining_labels}")

        # 如果移除了所有区域，保留最大的区域
        if len(remaining_labels) <= 1:  # 只剩下背景
            print("Debug: All areas were removed, keeping the largest area")
            max_area_idx = np.argmax(areas)
            largest_label = unique_labels[max_area_idx]
            new_labels = label_list.copy()
            # 只保留最大区域，其他设为背景
            for i in range(len(unique_labels)):
                if i != max_area_idx:
                    new_labels[new_labels == unique_labels[i]] = 0

        return new_labels
    else:
        print(f"Debug: Only {len(unique_labels)} labels found, no removal needed")
    return label_list


# Retrieves a list of labels and returns the centroid coordinates by calculating the average i and the averagre j index positions for the oring
def get_centroid(image):
    """
    计算图像的质心坐标
    
    参数:
        image (numpy.ndarray): 标记后的图像
    
    返回:
        list: 质心坐标 [i, j]
    """
    pixel = 0
    total_i = 0
    total_j = 0
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if image[i][j] > 0:  # 检查非零标签而不是仅仅检查1
                pixel += 1
                total_j += j
                total_i += i
    return [int(total_i / pixel),
            int(total_j / pixel)]  # 前景i位置的总和除以前景像素的计数得到平均i，j同理


# Retrieves a list of labels and calculates the coordinates for the top left and bottom right corners of the bounding box
def make_bounding_box(image):
    """
    计算边界框坐标
    
    参数:
        image (numpy.ndarray): 标记后的图像
    
    返回:
        list: 边界框坐标 [top, bottom, left, right]
    """
    bound_coords = [0 for i in range(4)]
    first_pixel = False  # 保险措施，确保在选择前景像素之前不会发生最小/最大坐标初始化
    for i in range(0, len(image)):
        for j in range(0, len(image[0])):
            if image[i][j] > 0:  # 检查非零标签而不是仅仅检查1
                if first_pixel == True:  # 检查这是否是第一个要处理的相关像素
                    if i < bound_coords[0]:
                        bound_coords[0] = i
                    elif i > bound_coords[1]:
                        bound_coords[1] = i
                    if j < bound_coords[2]:
                        bound_coords[2] = j
                    elif j > bound_coords[3]:
                        bound_coords[3] = j
                else:  # 如果是第一个找到的像素，它将用当前坐标设置i和j的最小值和最大值
                    first_pixel = True
                    bound_coords[0] = i
                    bound_coords[1] = i
                    bound_coords[2] = j
                    bound_coords[3] = j
    return bound_coords


# Retrieves an image, bounding box doorindates, a boolean result and returns the image with a bounding box around the ORing
def draw_bounding_box(image, bounding_box, result):
    """
    绘制边界框
    
    参数:
        image (numpy.ndarray): 输入图像
        bounding_box (list): 边界框坐标 [top, bottom, left, right]
        result (bool): 检测结果（True表示通过，False表示失败）
    
    返回:
        numpy.ndarray: 绘制了边界框的图像
    """
    pixel = (0, 0, 255)
    if result == True:
        pixel = (0, 255, 0)

    # 获取图像尺寸
    height, width = image.shape[:2]

    # 确保边界框坐标在图像范围内
    top = max(0, bounding_box[0] - 1)
    bottom = min(height, bounding_box[1] + 2)
    left = max(0, bounding_box[2] - 1)
    right = min(width - 1, bounding_box[3] + 1)  # 确保right < width

    # 绘制边界框
    image[top:bottom, left] = pixel
    image[top:bottom, right] = pixel
    image[top, left:right+1] = pixel
    image[bottom-1, left:right+1] = pixel
    return image


# Retrieves an image and centroid coordinate, then returns the average inne and outer edge radius values as a list with inner being 0 and outer being 1
def calculate_radius(image, centroid):
    """
    计算内外边缘半径
    
    参数:
        image (numpy.ndarray): 标记后的图像
        centroid (list): 质心坐标 [i, j]
    
    返回:
        list: 内外半径 [inner_radius, outer_radius]
    """
    radius = [0, 0]
    found = [False,
             False]  # 用于区分内外边缘半径值以及何时停止寻找它们

    # 动态获取图像边界，而不是硬编码220
    max_i, max_j = image.shape[0] - 1, image.shape[1] - 1

    # 质心向上
    for i in range(centroid[0] - 1, 0, -1):  # Loops from the centroid y position to the top of the image
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        # Handle multi-channel images by using np.all() for array comparisons
        current_pixel = image[i, centroid[1]]
        next_pixel = image[i + 1, centroid[1]]
        if np.all(current_pixel == 255) and np.all(
                next_pixel == 0):  # Checks if the current pixel is of a foreground colour and the previous pixel was background, then the inner edge has been found
            found[0] = True
        if np.all(current_pixel == 0) and np.all(
                next_pixel == 255):  # Checks if the current pixel is of a background colour and the previous pixel was foreground, then the outer edge has been reached
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break  # Breaks out of the loop after both edges have been found

    # The following 3 loops repeat this cycle as above but going in different directions

    # 质心向下
    for i in range(centroid[0], max_i):
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        current_pixel = image[i, centroid[1]]
        prev_pixel = image[i - 1, centroid[1]]
        if np.all(current_pixel == 255) and np.all(prev_pixel == 0):
            found[0] = True
        if np.all(current_pixel == 0) and np.all(prev_pixel == 255):
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # 质心向左
    for j in range(centroid[1], 0, -1):
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        current_pixel = image[centroid[0], j]
        next_pixel = image[centroid[0], j + 1]
        if np.all(current_pixel == 255) and np.all(next_pixel == 0):
            found[0] = True
        if np.all(current_pixel == 0) and np.all(next_pixel == 255):
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # 质心向右
    for j in range(centroid[1], max_j):
        if not found[0]:
            radius[0] += 1
        if not found[1]:
            radius[1] += 1
        current_pixel = image[centroid[0], j]
        prev_pixel = image[centroid[0], j - 1]
        if np.all(current_pixel == 255) and np.all(prev_pixel == 0):
            found[0] = True
        if np.all(current_pixel == 0) and np.all(prev_pixel == 255):
            found[1] = True
        if found[0] == True and found[1] == True:
            found = [False, False]
            break

    # 改进的半径计算：使用4个方向的平均值，并添加容差
    radius = [num / 4 for num in radius]
    # 移除减1操作，这可能导致半径过小
    # radius = [num - 1 for num in radius]

    # 确保半径不为负数
    radius = [max(1, num) for num in radius]

    return radius


# Retrieves x and y pixel coordinates to check, a and b coordinates for the center point of the circle and the radius, then returns true if x and y coordinates are inside the circle
def circle_center(x, y, a, b, radius):
    """
    检查点是否在圆内
    
    参数:
        x (int): 点的x坐标
        y (int): 点的y坐标
        a (int): 圆心的x坐标
        b (int): 圆心的y坐标
        radius (float): 圆的半径
    
    返回:
        bool: 如果点在圆内返回True，否则返回False
    """
    return ((x - a) ** 2) + ((y - b) ** 2) <= radius ** 2


# Retrieves an image, centroid coordinates, radius values and returns whether the ORing passes or fails
def oring_result(image, centroid, radius):
    """
    判断O型圈检测结果
    
    参数:
        image (numpy.ndarray): 标记后的图像
        centroid (list): 质心坐标 [i, j]
        radius (list): 内外半径 [inner_radius, outer_radius]
    
    返回:
        list: [检测结果(bool), 缺陷坐标列表, 期望形状图像]
    """
    # 增加容差，使检测更加宽松
    allowed_diff = 4  # 从2增加到4，允许更大的偏差
    shape_pixel = 255  # 分配给构造环形状的像素值，用于处理
    shape = image.copy()

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # 调用方法计算像素位置i和j是否在预期的环内
            if circle_center(j, i, centroid[1], centroid[0], radius[1]) and not circle_center(j, i, centroid[1],
                                                                                              centroid[0], radius[0]):
                shape[i, j] = shape_pixel
            else:
                shape[i, j] = 0

    # 存储任何缺陷像素坐标的列表
    faulty_coords = []

    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # 检查背景像素是否位于环应该在的位置，或者前景像素是否在环外找到
            # 通过使用np.all()处理多通道图像的数组比较
            current_pixel = image[i, j]
            current_shape = shape[i, j]
            if (np.all(current_pixel == 0) and np.all(current_shape == shape_pixel)) or (
                    np.all(current_pixel == 255) and np.all(current_shape == 0)):
                pixel_faulty = True
                # 从当前像素位置的-allowed_diff x,y坐标循环到+allowed_diff x,y
                for x in range(-allowed_diff, allowed_diff + 1):
                    for y in range(-allowed_diff, allowed_diff + 1):
                        if [x, y] != [0, 0]:
                            offset_x = i + x
                            offset_y = j + y
                            # 边界检查以避免在图像边界上出现IndexError
                            if offset_x < 0 or offset_x >= image.shape[0] or offset_y < 0 or offset_y >= image.shape[1]:
                                continue
                            if np.all(image[offset_x, offset_y] == 255):
                                pixel_faulty = False

                if pixel_faulty:
                    faulty_coords.append([i,
                                          j])  # 将其坐标添加到缺陷像素坐标列表中，以便重新着色缺陷区域

    # 改进的评判标准：允许一定比例的缺陷像素
    total_pixels = image.shape[0] * image.shape[1]
    defect_ratio = len(faulty_coords) / total_pixels

    # 如果缺陷像素比例小于1%，则认为是合格的
    # 这是一个严格的合格标准，适用于对质量要求极高的场景
    if defect_ratio < 0.01:
        return [True, faulty_coords, shape]

    # 如果缺陷像素比例小于5%，且缺陷像素数量少于100个，也认为是合格的
    # 这是一个相对宽松的标准，适用于对质量要求较高的场景
    if defect_ratio < 0.05 and len(faulty_coords) < 100:
        return [True, faulty_coords, shape]

    # 如果存在缺陷像素，则判定为不合格
    if len(faulty_coords) > 0:
        return [False, faulty_coords, shape]
    # 默认返回合格结果
    return [True, faulty_coords, shape]


# Retrieves an image and the expected ring shape
def draw_faulty_locations(image, expected_ring):
    """
    绘制缺陷位置
    
    参数:
        image (numpy.ndarray): 输入图像
        expected_ring (numpy.ndarray): 期望的环形形状
    
    返回:
        numpy.ndarray: 绘制了缺陷位置的图像
    """
    colour = np.array([0, 255, 0])  # 设置绘制颜色为绿色以修复区域
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            # 检查前景像素是否在预期环外，或者背景像素是否在预期环内
            if np.array_equal(expected_ring[i, j], np.array([255, 255, 255])) and np.array_equal(image[i, j], np.array(
                    [0, 0, 0])) or np.array_equal(expected_ring[i, j], np.array([0, 0, 0])) and np.array_equal(
                image[i, j], np.array([255, 255, 255])):
                image[i, j] = colour  # 将像素绘制为绿色
    return image


# Retrieves the title for the file name, the image, labels, centroid coords, bounding box coordinates around the ring, the processing time from the start as well as the total amount of passes and fails for the ORings
def start_process(title, image, labels, centroid, bounding_box, start, totals):
    """
    开始处理图像的主要函数
    
    参数:
        title (str): 图像标题
        image (numpy.ndarray): 输入图像
        labels (numpy.ndarray): 标记后的图像
        centroid (list): 质心坐标
        bounding_box (list): 边界框坐标
        start (float): 开始处理时间
        totals (list): 总的通过和失败计数
    
    返回:
        list: 更新后的通过和失败计数
    """
    drawn_image = image.copy()  # 创建原始图像的副本以绘制修复的像素
    drawn_image = paint_labels(drawn_image,
                               labels)  # Paints the labels of the components on the image which inverts the foreground and background colours
    radius = calculate_radius(drawn_image,
                              centroid)  # Calculates the 2 radius values which are the inner and outer edge of the ORing
    result = oring_result(drawn_image, centroid,
                          radius)  # Processes the image and returns true or false for whether it passed
    finish = time.time()  # Retrieves the time for when all the processing is complete
    results[2] = round((finish - start),
                       3)  # Adds the processing time for the current ring to the total processing time count rounded to 3 decimals

    # Check if the image is already multi-channel (RGB) or grayscale
    if len(drawn_image.shape) == 3 and drawn_image.shape[2] == 3:
        # Image is already RGB, no need to convert
        pass
    else:
        # Convert grayscale to RGB
        drawn_image = cv.cvtColor(drawn_image, cv.COLOR_GRAY2RGB)

    # Handle result[2] conversion if it's an image
    if isinstance(result[2], np.ndarray):
        if len(result[2].shape) == 3 and result[2].shape[2] == 3:
            # Already RGB
            pass
        else:
            # Convert to RGB
            result[2] = cv.cvtColor(result[2], cv.COLOR_GRAY2RGB)

    # Checks if the result for the current ORing has come back as true or false
    if result[0] == False:
        result_output = 'FAILED'
        colour = (0, 0, 255)
        results[1] += 1
        drawn_image = draw_faulty_locations(drawn_image, result[
            2])  # Draws in the areas that are expected but missing in green within the circle
    else:
        result_output = 'PASSED'
        colour = (0, 255, 0)
        results[0] += 1

    # Declares a font and line type variable
    font = cv.FONT_HERSHEY_SIMPLEX
    line = cv.LINE_AA

    drawn_image = draw_bounding_box(drawn_image, bounding_box, result[0])  # Draws the bounding box around the circle
    drawn_image = cv.putText(drawn_image, "Time: " + str(round(finish - start, 3)), (5, 10), font, 0.4, (255, 255, 255),
                             1, line)  # Paints the processing time text onto the image
    drawn_image = cv.putText(drawn_image, result_output, (170, 10), font, 0.4, colour, 1,
                             line)  # Prints the process result top right of the interface
    drawn_image = cv.putText(drawn_image, "Results: ", (5, 210), font, 0.4, (255, 255, 255), 1,
                             line)  # Prints results beside passed or failed for the results of tests
    drawn_image = cv.putText(drawn_image, "Passed: {}".format(results[0]), (80, 200), font, 0.3, (0, 255, 0), 1,
                             line)  # Prints the total number of passes for ORings
    drawn_image = cv.putText(drawn_image, "Failed: {}".format(results[1]), (80, 215), font, 0.3, (0, 0, 255), 1,
                             line)  # Prints the total number of fails for ORings

    # Optional for lines around the expected circle
    # radius = calculate_radius(image, centroid)
    # drawn_image = cv.circle(drawn_image, (centroid[1], centroid[0]), round(radius[0]), (125,125,125), 1)
    # drawn_image = cv.circle(drawn_image, (centroid[1], centroid[0]), round(radius[1]), (125,125,125), 1)

    cv.imshow(title, drawn_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return results


########################################################
#################### Process begins ####################
########################################################

if __name__ == '__main__':
    """
    主程序入口
    
    该部分负责读取O型圈图像，执行完整的图像处理流水线，
    包括阈值分割、形态学操作、连通分量标记、缺陷检测和结果可视化。
    """
    # Call the method to read in all the images, stores them in a list
    # images = read_images(directory)
    images = glob.glob(os.path.join(directory, '*.png'))  # 使用directory变量
    # Image counter
    imageCount = 1

    # Morphological structure which is used for erosion and dilation
    # 3x3的结构元素用于形态学操作（腐蚀和膨胀）
    morph_struct = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

    # Results list which contains the total number of ORings which have passed and the time for processing
    # results[0]: 通过的O型圈数量
    # results[1]: 失败的O型圈数量
    # results[2]: 总处理时间
    results = [0, 0, 0]

    # Loops through the list of oring images scanned
    for i in images:
        # image= cv.imread(i,0)
        # cv.imshow(i, image)
        # cv.waitKey(0)
        image = cv.imread(i)
        # Histogram plotting for the current image in the for loop
        plt.plot(image_hist(image))
        plt.show()

        start = time.time()
        image_threshold = threshold(image)
        cv.imshow('image_threshold', image_threshold)
        cv.waitKey(0)

        # 添加形态学处理来清理图像噪声
        # 使用闭运算（先膨胀后腐蚀）来填充小的黑色区域并去除小的白色噪声
        image_threshold = closing(image_threshold, morph_struct)
        cv.imshow('image_closing', image_threshold)
        cv.waitKey(0)
        cv.destroyAllWindows()

        image_labels = component_label(image_threshold)

        # 修复标签图像显示问题
        unique_labels = np.unique(image_labels)
        if len(unique_labels) > 1:
            # 将标签图像归一化到0-255范围以便显示
            display_labels = (image_labels * 255 // np.max(unique_labels)).astype(np.uint8)
            cv.imshow('component_label', display_labels)
        else:
            cv.imshow('component_label', image_labels)
        cv.waitKey(0)

        image_labels = remove_smallest_areas(image_labels)

        # 修复清理后标签图像显示问题
        unique_labels_cleaned = np.unique(image_labels)
        if len(unique_labels_cleaned) > 1:
            # 将标签图像归一化到0-255范围以便显示
            display_labels_cleaned = (image_labels * 255 // np.max(unique_labels_cleaned)).astype(np.uint8)
            cv.imshow('remove_smallest_areas', display_labels_cleaned)
        else:
            cv.imshow('remove_smallest_areas', image_labels)
        cv.waitKey(0)
        centroid = get_centroid(image_labels)
        bounding_box = make_bounding_box(image_labels)
        results = start_process(i.split('/')[-1], image_threshold, image_labels, centroid, bounding_box, start, results)

        # Print the processing statistics to the terminal
        print("---------- ORing Processing Results ----------")
        print("Time: {} seconds".format(results[2]))
        print("Passed ORings: {}".format(results[0]))
        print("Failed ORings: {}".format(results[1]))
        print("Images complete: {}".format(imageCount))
        imageCount += 1