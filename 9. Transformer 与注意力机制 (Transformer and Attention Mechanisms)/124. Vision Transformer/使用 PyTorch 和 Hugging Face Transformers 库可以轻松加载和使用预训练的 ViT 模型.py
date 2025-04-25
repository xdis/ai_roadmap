import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image

# 加载预训练模型和特征提取器
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# 加载并预处理图像
image = Image.open('cat.jpg')
inputs = feature_extractor(images=image, return_tensors="pt")

# 前向传播
outputs = model(**inputs)
logits = outputs.logits

# 获取预测结果
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])