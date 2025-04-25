# Pillow 图像处理库

Pillow是Python中强大的图像处理库，是PIL (Python Imaging Library) 的一个分支。它提供了广泛的图像处理功能，包括图像打开、创建、编辑和保存。

## 基本介绍

- **安装方法**：`pip install Pillow`
- **主要功能**：图像打开、格式转换、缩放、裁剪、旋转、滤镜应用等
- **支持格式**：JPEG、PNG、BMP、GIF、TIFF等常见图像格式

## 基础用法

### 1. 打开和显示图像

```python
from PIL import Image

# 打开图像
img = Image.open('example.jpg')

# 显示图像
img.show()

# 获取图像信息
print(f"图像格式: {img.format}")
print(f"图像大小: {img.size}")
print(f"图像模式: {img.mode}")
```

### 2. 图像保存与格式转换

```python
# 保存图像
img.save('new_image.png')

# 格式转换（例如 JPG 转 PNG）
img = Image.open('example.jpg')
img.save('example.png')
```

### 3. 图像调整大小

```python
# 调整图像大小
img = Image.open('example.jpg')
resized_img = img.resize((300, 200))  # 宽300像素，高200像素
resized_img.save('resized_image.jpg')

# 按比例缩放
width, height = img.size
new_width = 300
new_height = int(height * (new_width / width))
proportional_img = img.resize((new_width, new_height))
proportional_img.save('proportional_image.jpg')
```

### 4. 图像裁剪

```python
# 裁剪图像
# 参数是一个元组 (left, upper, right, lower)
img = Image.open('example.jpg')
cropped_img = img.crop((100, 100, 400, 400))
cropped_img.save('cropped_image.jpg')
```

### 5. 图像旋转和翻转

```python
# 旋转图像
img = Image.open('example.jpg')
rotated_img = img.rotate(45)  # 旋转45度
rotated_img.save('rotated_image.jpg')

# 水平翻转
flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
flipped_img.save('flipped_horizontal.jpg')

# 垂直翻转
flipped_img = img.transpose(Image.FLIP_TOP_BOTTOM)
flipped_img.save('flipped_vertical.jpg')
```

## 中级操作

### 1. 图像滤镜和增强

```python
from PIL import Image, ImageFilter, ImageEnhance

# 应用模糊滤镜
img = Image.open('example.jpg')
blurred = img.filter(ImageFilter.BLUR)
blurred.save('blurred_image.jpg')

# 边缘增强
edge_enhanced = img.filter(ImageFilter.EDGE_ENHANCE)
edge_enhanced.save('edge_enhanced.jpg')

# 调整亮度
enhancer = ImageEnhance.Brightness(img)
brightened = enhancer.enhance(1.5)  # 亮度提高50%
brightened.save('brightened.jpg')

# 调整对比度
enhancer = ImageEnhance.Contrast(img)
contrasted = enhancer.enhance(1.5)  # 对比度提高50%
contrasted.save('contrasted.jpg')
```

### 2. 绘制和添加文本

```python
from PIL import Image, ImageDraw, ImageFont

# 创建一个新的空白图像
img = Image.new('RGB', (400, 300), color='white')

# 创建绘图对象
draw = ImageDraw.Draw(img)

# 绘制形状
draw.rectangle([(50, 50), (200, 200)], outline='red', fill='blue')
draw.ellipse([(220, 50), (350, 150)], outline='green', fill='yellow')
draw.line([(50, 220), (350, 220)], fill='black', width=5)

# 添加文本
try:
    # 尝试加载系统字体
    font = ImageFont.truetype('arial.ttf', 24)
    draw.text((100, 250), "Hello, Pillow!", fill='black', font=font)
except IOError:
    # 如果找不到字体，使用默认字体
    draw.text((100, 250), "Hello, Pillow!", fill='black')

img.save('drawing_example.jpg')
```

### 3. 图像合并和拼接

```python
from PIL import Image

# 打开两个图像
img1 = Image.open('image1.jpg')
img2 = Image.open('image2.jpg')

# 确保两个图像大小相同
img2 = img2.resize(img1.size)

# 水平拼接
width, height = img1.size
merged = Image.new('RGB', (width * 2, height))
merged.paste(img1, (0, 0))
merged.paste(img2, (width, 0))
merged.save('horizontal_merged.jpg')

# 垂直拼接
merged = Image.new('RGB', (width, height * 2))
merged.paste(img1, (0, 0))
merged.paste(img2, (0, height))
merged.save('vertical_merged.jpg')
```

## 实际应用案例

### 1. 图像批处理

```python
import os
from PIL import Image, ImageEnhance

def batch_process_images(input_folder, output_folder, size=(800, 600), enhance_brightness=1.2):
    """
    批量处理图像：调整大小并增强亮度
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            
            # 打开图像
            img = Image.open(input_path)
            
            # 调整大小
            img = img.resize(size, Image.LANCZOS)
            
            # 增强亮度
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(enhance_brightness)
            
            # 保存处理后的图像
            output_path = os.path.join(output_folder, filename)
            img.save(output_path)
            
            print(f"处理完成: {filename}")

# 调用示例
# batch_process_images('input_images', 'output_images')
```

### 2. 创建图像缩略图

```python
from PIL import Image
import os

def create_thumbnails(input_folder, output_folder, size=(200, 200)):
    """
    为文件夹中的所有图像创建缩略图
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            
            # 创建缩略图文件名
            name, ext = os.path.splitext(filename)
            thumbnail_name = f"{name}_thumb{ext}"
            output_path = os.path.join(output_folder, thumbnail_name)
            
            # 创建缩略图
            img = Image.open(input_path)
            img.thumbnail(size)
            img.save(output_path)
            
            print(f"创建缩略图: {thumbnail_name}")

# 调用示例
# create_thumbnails('images', 'thumbnails')
```

## 总结

Pillow是Python中处理图像的重要库，提供了丰富的功能：

1. **基础操作**：打开、保存、调整大小、裁剪、旋转图像
2. **高级操作**：应用滤镜、增强图像、绘制图形、添加文本
3. **批处理能力**：能够批量处理大量图像

与OpenCV相比，Pillow更专注于图像处理的基本操作，而OpenCV则更侧重于计算机视觉算法。对于简单的图像处理任务，Pillow通常是更简单、更轻量级的选择。

## 常见问题解决

- **内存错误**：处理大图像时，可以使用`img.thumbnail()`而不是`img.resize()`来减少内存使用
- **图像质量**：保存JPEG图像时，可以设置质量参数：`img.save('output.jpg', quality=95)`
- **透明度处理**：处理带Alpha通道的PNG图像时，确保使用'RGBA'模式