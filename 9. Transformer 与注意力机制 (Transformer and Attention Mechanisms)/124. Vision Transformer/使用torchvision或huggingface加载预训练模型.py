import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

# 加载预训练模型
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()

# 预处理
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像
from PIL import Image
img = Image.open("example.jpg")
input_tensor = preprocess(img).unsqueeze(0)  # 添加batch维度

# 进行推理
with torch.no_grad():
    output = model(input_tensor)
    
# 获取预测类别
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)