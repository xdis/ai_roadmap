import os
import torch
import logging
import requests
import socket
import platform
import sys
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
            logging.FileHandler("bert_test_log.txt", mode="w", encoding="utf-8")  # 明确指定UTF-8编码
        ]
)
logger = logging.getLogger(__name__)

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
            logger.warning(f"无法创建默认缓存目录: {str(e)}")
            cache_writable = False
    
    return cache_writable

def test_bert_tokenizer_loading():
    """测试加载BERT tokenizer"""
    logger.info("="*50)
    logger.info("测试BERT Tokenizer加载")
    logger.info("="*50)
    
    from transformers import BertTokenizer
    
    # 创建本地缓存目录
    cache_dir = os.path.join(os.getcwd(), "bert_test_cache")
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
    for method in methods:
        logger.info(f"尝试方法: {method['name']}")
        try:
            tokenizer = method["func"]()
            logger.info(f"成功! 词表大小: {tokenizer.vocab_size}")
            succeeded = True
            break
        except Exception as e:
            logger.error(f"失败: {str(e)}")
    
    return succeeded, cache_dir

def test_bert_model_loading(cache_dir=None):
    """测试加载BERT模型"""
    logger.info("="*50)
    logger.info("测试BERT模型加载")
    logger.info("="*50)
    
    from transformers import BertForSequenceClassification
    
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "bert_test_cache")
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
    
    return succeeded

def show_recommendations(tokenizer_success, model_success, internet_ok, hf_ok, cache_ok):
    """显示建议"""
    logger.info("="*50)
    logger.info("测试结果和建议")
    logger.info("="*50)
    
    if tokenizer_success and model_success:
        logger.info("✓ 测试成功! BERT模型和Tokenizer都能够加载。")
        return
    
    logger.warning("× 测试失败! 无法完全加载BERT模型组件。")
    
    # 根据测试结果提供建议
    if not internet_ok:
        logger.info("问题: 互联网连接不可用")
        logger.info("建议:")
        logger.info("  1. 检查您的网络连接")
        logger.info("  2. 确认防火墙没有阻止Python访问互联网")
    
    if internet_ok and not hf_ok:
        logger.info("问题: 无法访问HuggingFace网站")
        logger.info("建议:")
        logger.info("  1. 检查您的网络是否允许访问huggingface.co")
        logger.info("  2. 如果在中国大陆，可能需要使用代理")
        logger.info("  3. 尝试使用镜像站点或预下载模型")
    
    if not cache_ok:
        logger.info("问题: 缓存目录权限问题")
        logger.info("建议:")
        logger.info("  1. 使用管理员/root权限运行脚本")
        logger.info("  2. 设置环境变量指定其他缓存目录:")
        logger.info("     - Windows: set TRANSFORMERS_CACHE=D:\\your\\cache\\dir")
        logger.info("     - Linux/Mac: export TRANSFORMERS_CACHE=/your/cache/dir")
    
    logger.info("\n其他通用建议:")
    logger.info("  1. 尝试使用更小的模型如 'distilbert-base-uncased'")
    logger.info("  2. 确保已安装最新版本的transformers库:")
    logger.info("     pip install --upgrade transformers")
    logger.info("  3. 手动预下载模型，然后使用local_files_only=True加载")
    logger.info("  4. 临时添加代理设置:")
    logger.info("     import os")
    logger.info("     os.environ['HTTP_PROXY'] = 'http://your-proxy:port'")
    logger.info("     os.environ['HTTPS_PROXY'] = 'http://your-proxy:port'")

def main():
    """主测试函数"""
    logger.info("开始BERT模型加载测试")
    
    # 1. 检查系统信息
    check_system_info()
    
    # 2. 检查网络连接
    internet_ok, hf_ok = check_internet_connection()
    
    # 3. 检查缓存目录
    cache_ok = check_cache_directory()
    
    # 4. 测试加载tokenizer
    tokenizer_success, cache_dir = test_bert_tokenizer_loading()
    
    # 5. 测试加载模型
    model_success = test_bert_model_loading(cache_dir)
    
    # 6. 显示建议
    show_recommendations(tokenizer_success, model_success, internet_ok, hf_ok, cache_ok)
    
    logger.info("测试完成!")
    logger.info(f"详细日志已保存至: {os.path.join(os.getcwd(), 'bert_test_log.txt')}")

if __name__ == "__main__":
    main()