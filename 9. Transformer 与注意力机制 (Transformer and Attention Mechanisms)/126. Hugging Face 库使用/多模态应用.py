from transformers import VisionTextDualEncoderModel, CLIPProcessor
import torch
from PIL import Image

# 1. 加载CLIP模型和处理器
model = VisionTextDualEncoderModel.from_pretrained("clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("clip-vit-base-patch32")

# 2. 准备图像和文本
image = Image.open("cat.jpg")
texts = ["一只猫", "一只狗", "一辆汽车", "一栋房子"]

# 3. 处理输入
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 4. 计算相似度
with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # 图像与文本的相似度分数
    probs = logits_per_image.softmax(dim=1)      # 将分数转换为概率
    
# 5. 显示结果
for text, prob in zip(texts, probs[0]):
    print(f"'{text}': {prob:.4f}")