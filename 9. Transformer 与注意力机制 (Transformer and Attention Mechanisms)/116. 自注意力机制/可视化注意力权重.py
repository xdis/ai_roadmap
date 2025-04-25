import torch
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(sentence, attention_matrix):
    """
    可视化注意力矩阵
    sentence: 单词列表
    attention_matrix: 形状为 [seq_len, seq_len] 的注意力权重矩阵
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, 
                xticklabels=sentence, 
                yticklabels=sentence, 
                cmap="YlGnBu", 
                annot=True)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title('Self-Attention Weights')
    plt.tight_layout()
    plt.show()

# 示例使用
words = ["小狗", "看见", "了", "一只", "猫"]
# 假设的注意力权重矩阵
attn = torch.softmax(torch.randn(5, 5), dim=-1).numpy()
visualize_attention(words, attn)