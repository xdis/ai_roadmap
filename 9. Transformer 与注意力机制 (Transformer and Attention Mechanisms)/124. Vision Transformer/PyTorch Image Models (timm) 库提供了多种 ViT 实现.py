import timm
import torch
from PIL import Image
import torchvision.transforms as transforms

# 加载预训练ViT模型
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 处理图像
img = Image.open('cat.jpg')
img_tensor = transform(img).unsqueeze(0)

# 预测
with torch.no_grad():
    output = model(img_tensor)

# 获取结果
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# 打印结果(需要ImageNet类别标签)
for i in range(top5_prob.size(0)):
    print(f"类别 {top5_catid[i]}: {top5_prob[i].item():.4f}")