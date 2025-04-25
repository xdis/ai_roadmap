from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

class VisualQA:
    def __init__(self, model_name="dandelin/vilt-b32-finetuned-vqa"):
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def answer_question(self, image_path, question):
        """回答关于图像的问题"""
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 处理输入
        inputs = self.processor(image, question, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = self.model.config.id2label[idx]
        
        return answer

# 使用示例
def test_visual_qa():
    vqa_system = VisualQA()
    
    # 示例图像路径(需要替换为实际图像)
    image_path = "example_image.jpg"
    
    questions = [
        "图片中有什么动物？",
        "这个人在做什么？",
        "图像中的场景是什么时间？"
    ]
    
    for question in questions:
        try:
            answer = vqa_system.answer_question(image_path, question)
            print(f"问题: {question}")
            print(f"答案: {answer}")
            print("-" * 50)
        except Exception as e:
            print(f"处理问题'{question}'时出错: {str(e)}")