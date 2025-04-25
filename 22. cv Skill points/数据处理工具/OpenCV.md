# OpenCV 基础教程

OpenCV（Open Source Computer Vision Library）是一个开源的计算机视觉库，广泛应用于图像处理、视频分析和机器视觉领域。

## 1. OpenCV 简介

OpenCV 提供了丰富的图像处理和计算机视觉算法，支持多种编程语言（如Python、C++、Java等），跨平台运行。

### 安装方法
```python
# 使用pip安装
pip install opencv-python

# 如果需要额外模块（如contrib）
pip install opencv-contrib-python
```

## 2. 基础操作

### 2.1 读取、显示和保存图像

```python
import cv2
import numpy as np

# 读取图像
img = cv2.imread('example.jpg')  # 读取彩色图像
# img = cv2.imread('example.jpg', 0)  # 读取灰度图像

# 显示图像
cv2.imshow('Image', img)
cv2.waitKey(0)  # 等待任意键继续
cv2.destroyAllWindows()  # 关闭所有窗口

# 保存图像
cv2.imwrite('saved_image.jpg', img)
```

### 2.2 图像基本信息

```python
# 获取图像尺寸
height, width, channels = img.shape  # 彩色图像
# height, width = img.shape  # 灰度图像

print(f"图像尺寸: {width} x {height}")
print(f"图像通道数: {channels}")  # RGB图像有3个通道
print(f"图像数据类型: {img.dtype}")  # 通常为uint8
```

## 3. 图像处理基础

### 3.1 颜色空间转换

```python
# BGR转灰度图
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# BGR转HSV
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR转RGB (OpenCV默认是BGR格式)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
```

### 3.2 图像缩放和旋转

```python
# 缩放图像
resized_img = cv2.resize(img, (300, 200))  # 指定宽高
resized_img2 = cv2.resize(img, None, fx=0.5, fy=0.5)  # 按比例缩放

# 旋转图像
rows, cols = img.shape[:2]
rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)  # 旋转45度
rotated_img = cv2.warpAffine(img, rotation_matrix, (cols, rows))
```

### 3.3 图像裁剪和添加边框

```python
# 裁剪图像
cropped_img = img[100:300, 200:400]  # [y1:y2, x1:x2]

# 添加边框
bordered_img = cv2.copyMakeBorder(
    img, 10, 10, 10, 10,  # 上下左右边框宽度
    cv2.BORDER_CONSTANT,  # 边框类型
    value=(0, 0, 255)  # 边框颜色 (BGR)
)
```

## 4. 图像处理技术

### 4.1 图像模糊/平滑处理

```python
# 高斯模糊
blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # (5,5)是kernel大小

# 中值滤波 (去除椒盐噪声效果好)
median_img = cv2.medianBlur(img, 5)

# 平均模糊
avg_img = cv2.blur(img, (5, 5))
```

### 4.2 边缘检测

```python
# Canny边缘检测
edges = cv2.Canny(img, 100, 200)  # 100和200是阈值

# Sobel边缘检测
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)  # x方向
sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)  # y方向
```

### 4.3 阈值处理

```python
# 简单阈值处理
_, thresh1 = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# 自适应阈值
adaptive_thresh = cv2.adaptiveThreshold(
    gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
```

## 5. 目标检测基础

### 5.1 轮廓检测

```python
# 轮廓检测
contours, _ = cv2.findContours(
    thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# 绘制轮廓
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
```

### 5.2 形状检测实例

```python
# 检测并标记矩形
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 根据轮廓面积过滤小噪点
    if cv2.contourArea(cnt) > 100:  
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
```

## 6. 特征检测与匹配

### 6.1 角点检测

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)

# 绘制角点
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
```

### 6.2 特征匹配示例

```python
# SIFT特征检测（需要opencv-contrib-python）
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_img, None)

# 绘制关键点
sift_img = cv2.drawKeypoints(img, keypoints, None)
```

## 7. 实际应用示例

### 7.1 人脸检测

```python
# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 检测人脸
faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

# 在图像中标记人脸
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

### 7.2 简单物体跟踪

```python
# 颜色范围定义(HSV)
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

# 创建蓝色物体的掩码
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# 找到蓝色物体的轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 绘制最大轮廓
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        
        # 标记物体中心
        cv2.circle(img, (cx, cy), 7, (0, 255, 0), -1)
```

## 8. 视频处理基础

```python
# 从摄像头读取视频
cap = cv2.VideoCapture(0)  # 0代表默认摄像头

# 检查是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取视频帧
while True:
    # 读取一帧
    ret, frame = cap.read()
    
    # 如果读取失败，跳出循环
    if not ret:
        print("无法获取视频帧")
        break
    
    # 处理帧 (例如转为灰度)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 显示结果
    cv2.imshow('Video', gray)
    
    # 按 'q' 键退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
```

## 总结

OpenCV 是计算机视觉领域非常强大的工具，提供了从基础图像处理到高级视觉算法的丰富功能。本指南涵盖了基础知识，但OpenCV的功能远不止于此。要深入学习，建议查阅官方文档和教程，并通过实际项目积累经验。

记住，OpenCV中图像是以numpy数组形式表示的，这让它能够与其他数据科学工具无缝集成。