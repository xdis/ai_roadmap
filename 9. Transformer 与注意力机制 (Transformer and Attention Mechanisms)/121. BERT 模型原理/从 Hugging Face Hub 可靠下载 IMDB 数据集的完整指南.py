import os
import time
import logging
import urllib.request
import tarfile
import socket
from typing import Dict, Any, Optional
from itertools import islice

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_and_load_imdb(sample_size=None):
    """下载并加载IMDB数据集"""
    logger.info("开始直接下载IMDB数据集...")
    
    # 创建数据目录
    os.makedirs("./imdb_data", exist_ok=True)
    
    # 下载地址 (从官方来源)
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./imdb_data/aclImdb_v1.tar.gz"
    
    # 如果文件不存在，则下载
    if not os.path.exists(save_path):
        logger.info(f"正在从 {url} 下载IMDB数据集...")
        try:
            urllib.request.urlretrieve(url, save_path)
            logger.info(f"下载成功，保存到 {save_path}")
        except Exception as e:
            logger.error(f"下载失败: {str(e)}")
            return create_mock_imdb_dataset()
    else:
        logger.info(f"找到已下载的数据集: {save_path}")
    
    # 解压数据
    extract_dir = "./imdb_data/aclImdb"
    if not os.path.exists(extract_dir):
        logger.info("正在解压数据...")
        with tarfile.open(save_path, 'r:gz') as tar:
            tar.extractall(path="./imdb_data/")
        logger.info("解压完成")
    else:
        logger.info(f"找到已解压的数据: {extract_dir}")
    
    # 处理数据
    logger.info("正在处理数据...")
    train_pos_dir = os.path.join(extract_dir, "train", "pos")
    train_neg_dir = os.path.join(extract_dir, "train", "neg")
    test_pos_dir = os.path.join(extract_dir, "test", "pos")
    test_neg_dir = os.path.join(extract_dir, "test", "neg")
    
    # 读取文本文件
    def read_text_files(directory, limit=None):
        texts = []
        files = sorted(os.listdir(directory))
        if limit:
            files = files[:limit]
            
        for filename in files:
            if filename.endswith(".txt"):
                with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                    texts.append(file.read())
        return texts
    
    # 根据样本大小限制数据量
    per_class_limit = None
    if sample_size:
        per_class_limit = sample_size // 4  # 平均分配给4个类别
    
    # 创建数据集
    train_pos = read_text_files(train_pos_dir, per_class_limit)
    train_neg = read_text_files(train_neg_dir, per_class_limit)
    test_pos = read_text_files(test_pos_dir, per_class_limit)
    test_neg = read_text_files(test_neg_dir, per_class_limit)
    
    train_texts = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)
    
    test_texts = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)
    
    # 转换为 DatasetDict
    try:
        from datasets import Dataset, DatasetDict
        
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        logger.info(f"成功创建数据集! 训练集: {len(train_dataset)} 样本, 测试集: {len(test_dataset)} 样本")
        return dataset_dict
    
    except ImportError:
        logger.warning("未安装datasets库，返回原始数据")
        return {
            "train": {"text": train_texts, "label": train_labels},
            "test": {"text": test_texts, "label": test_labels}
        }

def create_mock_imdb_dataset():
    """创建模拟的IMDB数据集"""
    try:
        from datasets import Dataset, DatasetDict
        
        # 创建简单的模拟数据
        train_data = {
            'text': [
                "This movie was fantastic! I really enjoyed it.",
                "The film was terrible and boring.",
                "Great acting, compelling story, highly recommend!",
                "Waste of time and money, very disappointing.",
                "Excellent performances by the entire cast."
            ],
            'label': [1, 0, 1, 0, 1]  # 1=positive, 0=negative
        }
        
        test_data = {
            'text': [
                "I didn't like the plot and the ending was predictable.",
                "One of the best films I've seen this year.",
                "The special effects were good but the story was weak.",
                "A masterpiece of cinema, absolutely loved it!",
                "So bad I almost walked out of the theater."
            ],
            'label': [0, 1, 0, 1, 0]
        }
        
        # 创建数据集
        train_dataset = Dataset.from_dict(train_data)
        test_dataset = Dataset.from_dict(test_data)
        
        # 返回数据集字典
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    except ImportError:
        logger.warning("未安装datasets库，返回原始字典")
        return {
            'train': {'text': ["Sample positive review", "Sample negative review"], 
                      'label': [1, 0]},
            'test': {'text': ["Another positive review", "Another negative review"], 
                     'label': [1, 0]}
        }

def check_internet_connection():
    """检查网络连接"""
    try:
        # 尝试连接到Google的DNS服务器
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def load_imdb_dataset(
    max_retries: int = 2, 
    use_offline: bool = False,
    cache_dir: Optional[str] = None,
    use_auth_token: bool = False,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    可靠地获取IMDB数据集
    """
    # 首先确保必要的库已正确安装
    try:
        import datasets
        logger.info(f"使用datasets库版本: {datasets.__version__}")
    except ImportError:
        logger.warning("找不到datasets库，使用直接下载方法")
        return download_and_load_imdb(sample_size)
    
    # 检查网络连接
    if not check_internet_connection():
        logger.warning("网络连接失败，使用直接下载方法")
        return download_and_load_imdb(sample_size)
    
    # 如果已经尝试过Hugging Face方法且失败，直接使用下载方法
    logger.info("使用直接下载方法获取IMDB数据集")
    return download_and_load_imdb(sample_size)

# 示例用法
if __name__ == "__main__":
    # 加载数据集
    imdb_dataset = load_imdb_dataset(
        sample_size=5000  # 只加载5000个样本用于测试
    )
    
    # 打印数据集信息
    print(f"\n数据集加载完成!")
    
    if hasattr(imdb_dataset, 'keys'):
        print(f"训练集大小: {len(imdb_dataset['train'])}")
        print(f"测试集大小: {len(imdb_dataset['test'])}")
        
        # 打印几个示例
        print("\n示例数据:")
        for i in range(min(3, len(imdb_dataset['train']))):
            if hasattr(imdb_dataset['train'], '__getitem__'):
                text = imdb_dataset['train'][i]['text']
                label = "正面评价" if imdb_dataset['train'][i]['label'] == 1 else "负面评价"
                print(f"[{label}] {text[:100]}..." if len(text) > 100 else f"[{label}] {text}")
    else:
        print("数据集格式不符合预期，可能是低级别表示。")