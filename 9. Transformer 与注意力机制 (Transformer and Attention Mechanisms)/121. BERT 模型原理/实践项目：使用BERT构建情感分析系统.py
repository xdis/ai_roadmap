import os
import logging
import urllib.request
import tarfile
import socket
import torch
import numpy as np
import pandas as pd
import platform
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertTokenizerFast, BertForSequenceClassification
from torch.optim import AdamW
from typing import Dict, Any, Optional
from itertools import islice
import json
import requests
from pathlib import Path

# 设置脚本目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 设置日志记录
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # 使用stdout并确保编码正确
        logging.FileHandler("bert_sentiment_log.txt", mode="w", encoding="utf-8")  # 明确指定UTF-8编码
    ]
)
logger = logging.getLogger(__name__)

# ======== 系统检查 ========

def check_system_info():
    """检查系统信息"""
    logger.info("="*50)
    logger.info("系统环境信息")
    logger.info("="*50)
    logger.info(f"操作系统: {platform.system()} {platform.version()}")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    try:
        import transformers
        logger.info(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        logger.error("Transformers库未安装")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info("="*50)

def check_internet_connection():
    """检查网络连接"""
    logger.info("检查网络连接...")
    
    # 检查基本连接
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        logger.info("基本网络连接: 成功")
        basic_connection = True
    except OSError:
        logger.warning("基本网络连接: 失败 (无法连接到8.8.8.8)")
        basic_connection = False
    
    # 检查访问huggingface.co
    try:
        response = requests.get("https://huggingface.co", timeout=5)
        logger.info(f"访问HuggingFace: 成功 (状态码: {response.status_code})")
        hf_connection = True
    except Exception as e:
        logger.warning(f"访问HuggingFace: 失败 ({str(e)})")
        hf_connection = False
    
    return basic_connection, hf_connection

def check_cache_directory():
    """检查缓存目录"""
    # 获取默认缓存目录
    default_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    
    # 检查环境变量缓存目录
    env_cache = os.environ.get("TRANSFORMERS_CACHE", "未设置")
    
    logger.info("检查缓存目录...")
    logger.info(f"默认缓存目录: {default_cache}")
    logger.info(f"环境变量缓存目录: {env_cache}")
    
    # 检查默认缓存目录的权限
    if os.path.exists(default_cache):
        logger.info(f"默认缓存目录存在")
        try:
            test_file = os.path.join(default_cache, "test_write.tmp")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            logger.info("默认缓存目录写入权限: 成功")
            cache_writable = True
        except Exception as e:
            logger.warning(f"默认缓存目录写入权限: 失败 ({str(e)})")
            cache_writable = False
    else:
        logger.info("默认缓存目录不存在")
        try:
            os.makedirs(default_cache, exist_ok=True)
            logger.info("成功创建默认缓存目录")
            cache_writable = True
        except Exception as e:
            logger.warning(f"无法创建默认缓存目录: {str(e)})")
            cache_writable = False
    
    return cache_writable

# ======== 数据获取部分 ========

def download_and_load_imdb(sample_size=None):
    """下载并加载IMDB数据集"""
    logger.info("开始直接下载IMDB数据集...")
    
    # 创建数据目录
    os.makedirs("./imdb_data", exist_ok=True)
    
    # 下载地址 (从官方来源)
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    save_path = "./imdb_data/aclImdb_v1.tar.gz"
    
    # 如果文件不存在或文件大小不正确，则下载
    file_size_threshold = 80 * 1024 * 1024  # 80MB (实际文件约84MB)
    if not os.path.exists(save_path) or os.path.getsize(save_path) < file_size_threshold:
        if os.path.exists(save_path):
            logger.warning(f"检测到文件大小不正确，重新下载: {save_path}")
            os.remove(save_path)
            
        logger.info(f"正在从 {url} 下载IMDB数据集...")
        try:
            def report_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                percent = min(percent, 100)
                print(f"\r下载进度: {percent}% [{count*block_size}/{total_size} bytes]", end='', flush=True)
                
            urllib.request.urlretrieve(url, save_path, reporthook=report_progress)
            print()  # 换行
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
        try:
            with tarfile.open(save_path, 'r:gz') as tar:
                tar.extractall(path="./imdb_data/")
            logger.info("解压完成")
        except EOFError as e:
            logger.error(f"解压失败，文件可能已损坏: {str(e)}")
            logger.info("删除损坏的文件并重试...")
            os.remove(save_path)
            return download_and_load_imdb(sample_size)
        except Exception as e:
            logger.error(f"解压出错: {str(e)}")
            return create_mock_imdb_dataset()
    else:
        logger.info(f"找到已解压的数据: {extract_dir}")
    
    # 处理数据
    logger.info("正在处理数据...")
    train_pos_dir = os.path.join(extract_dir, "train", "pos")
    train_neg_dir = os.path.join(extract_dir, "train", "neg")
    test_pos_dir = os.path.join(extract_dir, "test", "pos")
    test_neg_dir = os.path.join(extract_dir, "test", "neg")
    
    # 检查目录是否存在
    required_dirs = [train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir]
    if not all(os.path.exists(d) for d in required_dirs):
        logger.error(f"数据目录结构不完整，可能解压有问题")
        return create_mock_imdb_dataset()
    
    # 读取文本文件
    def read_text_files(directory, limit=None):
        texts = []
        files = sorted(os.listdir(directory))
        if limit:
            files = files[:limit]
            
        for filename in files:
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                        texts.append(file.read())
                except Exception as e:
                    logger.warning(f"读取文件 {filename} 时出错: {str(e)}")
                    texts.append("Error reading this review.")
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
    
    logger.info(f"数据集加载成功! 训练: {len(train_texts)} 样本, 测试: {len(test_texts)} 样本")
    
    # 转换为字典格式
    dataset_dict = {
        "train": {"text": train_texts, "label": train_labels},
        "test": {"text": test_texts, "label": test_labels}
    }
    
    # 如果可能，转换为Dataset对象
    try:
        from datasets import Dataset, DatasetDict
        
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})
        
        dataset_dict = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })
        
        logger.info("成功创建HuggingFace Dataset对象")
    except ImportError:
        logger.warning("未安装datasets库，保持字典格式")
    
    return dataset_dict

def create_mock_imdb_dataset():
    """创建模拟的IMDB数据集"""
    logger.info("创建模拟IMDB数据集...")
    
    # 创建简单的模拟数据
    train_data = {
        'text': [
            "This movie was fantastic! I really enjoyed it and would recommend it to anyone looking for a great film experience.",
            "The film was terrible and boring. I couldn't wait for it to end and was checking my watch every five minutes.",
            "Great acting, compelling story, highly recommend! The director did an amazing job bringing these characters to life.",
            "Waste of time and money, very disappointing. The plot had holes big enough to drive a truck through.",
            "Excellent performances by the entire cast. Everyone delivered their lines with conviction and the chemistry was palpable."
        ],
        'label': [1, 0, 1, 0, 1]  # 1=positive, 0=negative
    }
    
    test_data = {
        'text': [
            "I didn't like the plot and the ending was predictable. You could see the twist coming from a mile away.",
            "One of the best films I've seen this year. I was captivated from beginning to end.",
            "The special effects were good but the story was weak. Style over substance throughout the entire movie.",
            "A masterpiece of cinema, absolutely loved it! Each scene was carefully crafted and meaningful.",
            "So bad I almost walked out of the theater. Save your money and watch something else instead."
        ],
        'label': [0, 1, 0, 1, 0]
    }
    
    # 尝试使用Dataset对象
    try:
        from datasets import Dataset, DatasetDict
        
        train_dataset = Dataset.from_dict(train_data)
        test_dataset = Dataset.from_dict(test_data)
        
        return DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })
    except ImportError:
        return {
            'train': train_data,
            'test': test_data
        }

def get_imdb_dataset(sample_size=None):
    """获取IMDB数据集的主函数"""
    
    # 检查网络连接
    basic_connection, hf_connection = check_internet_connection()
    
    if not basic_connection:
        logger.warning("网络连接失败，使用直接下载或本地缓存")
    
    # 尝试从Hugging Face Hub加载
    if hf_connection:
        try:
            from datasets import load_dataset
            logger.info("尝试从Hugging Face Hub加载IMDB数据集...")
            
            # 设置环境变量使用镜像站点
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                logger.info("已设置HuggingFace镜像站点")
            
            imdb_dataset = load_dataset("imdb", trust_remote_code=True)
            logger.info("成功从Hugging Face加载IMDB数据集")
            
            # 如果需要采样
            if sample_size:
                logger.info(f"采样{sample_size}条数据...")
                train_sample = imdb_dataset['train'].select(range(min(sample_size, len(imdb_dataset['train']))))
                test_sample = imdb_dataset['test'].select(range(min(sample_size//10, len(imdb_dataset['test']))))
                imdb_dataset = {'train': train_sample, 'test': test_sample}
                
            return imdb_dataset
            
        except Exception as e:
            logger.warning(f"从Hugging Face加载失败: {str(e)}")
    
    logger.info("尝试直接下载IMDB数据集...")
    return download_and_load_imdb(sample_size)

# ======== 模型加载部分（增强版）========

def create_simple_tokenizer():
    """创建一个简单的tokenizer用于离线环境"""
    logger.info("创建简单的离线tokenizer...")
    
    # 确保模型缓存目录存在
    model_dir = os.path.join(script_dir, "bert_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建词表文件
    vocab_path = os.path.join(model_dir, "vocab.txt")
    
    # 只有在文件不存在时创建词表
    if not os.path.exists(vocab_path):
        logger.info("创建基础BERT词表文件...")
        
        # 创建一个简化版的词表（实际BERT词表有30522个词）
        # 这里只创建最基础的几百个词，足够处理简单的文本
        vocab_words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        # 添加常用的英文单词
        common_words = [
            "the", "of", "and", "a", "to", "in", "is", "you", "that", "it", "he",
            "was", "for", "on", "are", "with", "as", "i", "his", "they", "be",
            "at", "one", "have", "this", "from", "or", "had", "by", "not", "word",
            "but", "what", "some", "we", "can", "out", "other", "were", "all", "there",
            "when", "up", "use", "your", "how", "said", "an", "each", "she", "which",
            "do", "their", "time", "if", "will", "way", "about", "many", "then", "them",
            "would", "write", "like", "so", "these", "her", "long", "make", "thing", "see",
            "him", "two", "has", "look", "more", "day", "could", "go", "come", "did",
            "number", "sound", "no", "most", "people", "my", "over", "know", "water", "than",
            "call", "first", "who", "may", "down", "side", "been", "now", "find", "any",
            "new", "work", "part", "take", "get", "place", "made", "live", "where", "after",
            "back", "little", "only", "round", "man", "year", "came", "show", "every", "good",
            "me", "give", "our", "under", "name", "very", "through", "just", "form", "sentence",
            "great", "think", "say", "help", "low", "line", "differ", "turn", "cause", "much"
        ]
        
        vocab_words.extend(common_words)
        
        # 添加电影评论中常见的单词
        movie_words = [
            "movie", "film", "scene", "story", "character", "director", "actor", "actress",
            "plot", "acting", "performance", "cinema", "theater", "production", "role",
            "script", "audience", "review", "critic", "entertainment", "drama", "comedy",
            "action", "thriller", "horror", "sci-fi", "fantasy", "romance", "adventure",
            "documentary", "cast", "filming", "visual", "effects", "soundtrack", "score",
            "dialogue", "screenplay", "cinematography", "editing", "sequel", "prequel",
            "series", "recommend", "rating", "star", "award", "oscar", "boring", "exciting",
            "disappointing", "amazing", "terrible", "excellent", "brilliant", "worst", "best",
            "favorite", "waste", "money", "ticket", "watch", "saw", "seen", "theater"
        ]
        
        vocab_words.extend(movie_words)
        
        # 添加情感分析相关的词汇
        sentiment_words = [
            "good", "bad", "great", "awful", "excellent", "terrible", "wonderful", "horrible",
            "like", "dislike", "love", "hate", "enjoy", "disappointed", "satisfied", "unsatisfied",
            "recommend", "avoid", "positive", "negative", "perfect", "worst", "best", "better",
            "worse", "amazing", "fantastic", "poor", "masterpiece", "disaster", "brilliant", "mediocre",
            "outstanding", "awful", "superb", "disappointing", "stunning", "boring", "exciting", "dull",
            "engaging", "tedious", "captivating", "tiresome", "gripping", "uninteresting", "compelling",
            "ordinary", "exceptional", "average", "impressive", "unimpressive", "strong", "weak"
        ]
        
        vocab_words.extend(sentiment_words)
        
        # 移除重复项并排序
        vocab_words = sorted(list(set(vocab_words)))
        
        # 确保特殊标记在词表的开头
        for special_token in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]:
            if special_token in vocab_words and vocab_words.index(special_token) != 0:
                vocab_words.remove(special_token)
                
        vocab_words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + vocab_words
        
        # 写入词表文件
        with open(vocab_path, "w", encoding="utf-8") as f:
            for word in vocab_words:
                f.write(f"{word}\n")
        
        logger.info(f"词表已创建，包含 {len(vocab_words)} 个词汇")
                
    # 创建配置文件
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        logger.info("创建BERT配置文件...")
        config = {
            "vocab_size": 1000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 2,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-12,
            "classifier_dropout": None,
            "model_type": "bert"
        }
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"配置文件已创建: {config_path}")
    
    # 创建tokenizer对象
    tokenizer = BertTokenizerFast(
        vocab_file=vocab_path,
        do_lower_case=True,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]"
    )
    
    return tokenizer

def test_bert_tokenizer_loading():
    """测试加载BERT tokenizer"""
    logger.info("="*50)
    logger.info("测试BERT Tokenizer加载")
    logger.info("="*50)
    
    # 创建本地缓存目录
    cache_dir = os.path.join(script_dir, "bert_test_cache")
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"使用本地缓存目录: {cache_dir}")
    
    # 测试方法列表
    methods = [
        {
            "name": "标准方式加载",
            "func": lambda: BertTokenizer.from_pretrained('bert-base-uncased')
        },
        {
            "name": "使用本地缓存目录",
            "func": lambda: BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir)
        },
        {
            "name": "使用本地缓存并强制下载",
            "func": lambda: BertTokenizer.from_pretrained('bert-base-uncased', cache_dir=cache_dir, force_download=True)
        },
        {
            "name": "加载小型BERT模型",
            "func": lambda: BertTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2', cache_dir=cache_dir)
        }
    ]
    
    # 测试各种方法
    succeeded = False
    tokenizer = None
    for method in methods:
        logger.info(f"尝试方法: {method['name']}")
        try:
            tokenizer = method["func"]()
            logger.info(f"成功! 词表大小: {tokenizer.vocab_size}")
            succeeded = True
            break
        except Exception as e:
            logger.error(f"失败: {str(e)}")
    
    return succeeded, tokenizer, cache_dir

def test_bert_model_loading(cache_dir=None):
    """测试加载BERT模型"""
    logger.info("="*50)
    logger.info("测试BERT模型加载")
    logger.info("="*50)
    
    if cache_dir is None:
        cache_dir = os.path.join(script_dir, "bert_test_cache")
        os.makedirs(cache_dir, exist_ok=True)
    
    # 测试方法列表
    methods = [
        {
            "name": "标准方式加载",
            "func": lambda: BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        },
        {
            "name": "使用本地缓存目录",
            "func": lambda: BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, cache_dir=cache_dir)
        },
        {
            "name": "加载小型BERT模型",
            "func": lambda: BertForSequenceClassification.from_pretrained('google/bert_uncased_L-2_H-128_A-2', num_labels=2, cache_dir=cache_dir)
        },
        {
            "name": "离线加载已下载的模型",
            "func": lambda: BertForSequenceClassification.from_pretrained(cache_dir, num_labels=2, local_files_only=True) 
                            if os.path.exists(os.path.join(cache_dir, "config.json")) else None
        }
    ]
    
    # 测试各种方法
    succeeded = False
    model = None
    for method in methods:
        logger.info(f"尝试方法: {method['name']}")
        try:
            model = method["func"]()
            if model is not None:
                logger.info(f"成功! 模型类型: {type(model).__name__}")
                succeeded = True
                break
        except Exception as e:
            logger.error(f"失败: {str(e)}")
    
    return succeeded, model

# ======== 模型训练部分 ========

class ReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len=128):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        
        encoding = self.tokenizer(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # 确保所有预期的键都存在
        if 'token_type_ids' not in encoding:
            encoding['token_type_ids'] = torch.zeros_like(encoding['input_ids'])
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        reviews=df.text.to_numpy(),
        targets=df.sentiment.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    losses = []
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        targets = batch['targets'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=targets
        )
        
        loss = outputs.loss
        losses.append(loss.item())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())
    
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def run_sentiment_analysis():
    """运行情感分析完整流程"""
    # 设置环境变量使用镜像站点(中国大陆用户友好)
    if 'HF_ENDPOINT' not in os.environ:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 检查系统环境
    check_system_info()
    
    # 获取数据集
    dataset = get_imdb_dataset(sample_size=1000)  # 减少样本数，仅用于测试
    
    # 准备数据框
    if hasattr(dataset, 'keys') and hasattr(dataset['train'], 'to_pandas'):
        # 处理HuggingFace Dataset对象
        train_df = dataset['train'].to_pandas()
        val_df = dataset['test'].to_pandas()
        # 确保列名正确
        if 'text' not in train_df.columns and 'text' in dataset['train'].column_names:
            train_df = train_df.rename(columns={dataset['train'].column_names[0]: 'text'})
        if 'label' not in train_df.columns and 'label' in dataset['train'].column_names:
            train_df = train_df.rename(columns={dataset['train'].column_names[1]: 'label'})
        # 对验证集做同样处理
        if 'text' not in val_df.columns and 'text' in dataset['test'].column_names:
            val_df = val_df.rename(columns={dataset['test'].column_names[0]: 'text'})
        if 'label' not in val_df.columns and 'label' in dataset['test'].column_names:
            val_df = val_df.rename(columns={dataset['test'].column_names[1]: 'label'})
    elif hasattr(dataset, 'keys'):
        # 处理字典格式
        train_data = {
            'text': dataset['train']['text'],
            'sentiment': dataset['train']['label']
        }
        val_data = {
            'text': dataset['test']['text'],
            'sentiment': dataset['test']['label']
        }
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
    else:
        # 数据格式不识别，创建模拟数据
        logger.error("数据格式无法识别，使用模拟数据")
        mock_dataset = create_mock_imdb_dataset()
        train_data = {
            'text': mock_dataset['train']['text'],
            'sentiment': mock_dataset['train']['label']
        }
        val_data = {
            'text': mock_dataset['test']['text'],
            'sentiment': mock_dataset['test']['label']
        }
        train_df = pd.DataFrame(train_data)
        val_df = pd.DataFrame(val_data)
    
    # 确保列名为sentiment
    if 'label' in train_df.columns and 'sentiment' not in train_df.columns:
        train_df = train_df.rename(columns={'label': 'sentiment'})
    if 'label' in val_df.columns and 'sentiment' not in val_df.columns:
        val_df = val_df.rename(columns={'label': 'sentiment'})
    
    logger.info(f"训练数据: {len(train_df)} 样本")
    logger.info(f"验证数据: {len(val_df)} 样本")
    
    # 使用测试脚本的方法加载tokenizer和模型
    logger.info("开始加载BERT模型和tokenizer...")
    tokenizer_success, tokenizer, cache_dir = test_bert_tokenizer_loading()
    
    # 如果加载失败，回退到简单tokenizer
    if not tokenizer_success or tokenizer is None:
        logger.warning("无法加载预训练tokenizer，使用简单tokenizer替代")
        tokenizer = create_simple_tokenizer()
    
    # 加载模型
    model_success, model = test_bert_model_loading(cache_dir)
    
    # 如果模型加载失败，抛出错误
    if not model_success or model is None:
        raise RuntimeError("无法加载BERT模型，请检查网络连接或使用代理")
    
    # 数据加载器
    try:
        train_data_loader = create_data_loader(train_df, tokenizer, 128, 16)
        val_data_loader = create_data_loader(val_df, tokenizer, 128, 16)
    except Exception as e:
        logger.error(f"创建数据加载器失败: {str(e)}")
        logger.info("尝试使用更小的batch size...")
        train_data_loader = create_data_loader(train_df, tokenizer, 128, 8)
        val_data_loader = create_data_loader(val_df, tokenizer, 128, 8)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    model = model.to(device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # 训练循环
    best_accuracy = 0
    epochs = 2  # 减少轮数，仅用于测试
    
    logger.info("开始训练...")
    for epoch in range(epochs):
        logger.info(f'Epoch {epoch + 1}/{epochs}')
        train_loss = train_epoch(model, train_data_loader, optimizer, device)
        logger.info(f'Train loss: {train_loss}')
        
        accuracy, report = eval_model(model, val_data_loader, device)
        logger.info(f'Val Accuracy: {accuracy}')
        logger.info(f'分类报告:\n{report}')
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = accuracy
            logger.info(f'保存模型到 best_model_state.bin')
            
    logger.info(f'最佳准确率: {best_accuracy}')
    logger.info("情感分析模型训练完成!")

if __name__ == '__main__':
    logger.info("开始BERT情感分析系统")
    
    # 检查可用的框架
    try:
        import torch
        logger.info(f"PyTorch version {torch.__version__} available.")
    except ImportError:
        logger.error("PyTorch not found. Please install PyTorch.")
        exit(1)
    
    # 运行情感分析
    run_sentiment_analysis()