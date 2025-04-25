from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# 1. 加载图像描述生成模型
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# 2. 加载图像
image = Image.open("example.jpg")

# 3. 处理图像
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

# 4. 生成描述
output_ids = model.generate(
    pixel_values,
    max_length=16,
    num_beams=4,
    early_stopping=True
)

# 5. 解码输出
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"图像描述: {caption}")