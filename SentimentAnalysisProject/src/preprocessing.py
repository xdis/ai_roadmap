"""
文本预处理模块：实现文本清洗、分词、词干提取和词形还原等功能
"""
import re
import string
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union, Optional

# 下载必要的NLTK资源
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# 加载spacy模型
try:
    nlp = spacy.load('en_core_web_sm')
    SPACY_AVAILABLE = True
except OSError:
    print("spaCy模型'en_core_web_sm'不可用。将使用NLTK作为备选方案。")
    SPACY_AVAILABLE = False
    # 不再尝试自动下载，因为可能会超时

# 获取英文停用词
STOP_WORDS = set(stopwords.words('english'))


class TextPreprocessor:
    """文本预处理类，提供多种文本清洗和标准化方法"""
    
    def __init__(self, remove_stopwords: bool = True, 
                 stemming: bool = False, 
                 lemmatization: bool = True,
                 min_word_length: int = 2):
        """
        初始化文本预处理器
        
        Args:
            remove_stopwords: 是否删除停用词
            stemming: 是否进行词干提取
            lemmatization: 是否进行词形还原
            min_word_length: 保留词的最小长度
        """
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.min_word_length = min_word_length
        
        # 初始化词干提取器和词形还原器
        if stemming:
            self.stemmer = PorterStemmer()
        if lemmatization:
            self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text: str) -> str:
        """
        基本文本清洗：移除URL、HTML标签、特殊字符等
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 转换为小写
        text = text.lower()
        
        # 移除URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 移除数字
        text = re.sub(r'\d+', '', text)
        
        # 移除标点符号
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词后的词列表
        """
        return word_tokenize(text)
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        从词列表中移除停用词
        
        Args:
            tokens: 词列表
            
        Returns:
            移除停用词后的词列表
        """
        return [token for token in tokens if token.lower() not in STOP_WORDS]
    
    def apply_stemming(self, tokens: List[str]) -> List[str]:
        """
        对词列表应用词干提取
        
        Args:
            tokens: 词列表
            
        Returns:
            词干提取后的词列表
        """
        return [self.stemmer.stem(token) for token in tokens]
    
    def apply_lemmatization(self, tokens: List[str]) -> List[str]:
        """
        对词列表应用词形还原
        
        Args:
            tokens: 词列表
            
        Returns:
            词形还原后的词列表
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def filter_short_words(self, tokens: List[str]) -> List[str]:
        """
        过滤掉过短的词
        
        Args:
            tokens: 词列表
            
        Returns:
            过滤后的词列表
        """
        return [token for token in tokens if len(token) >= self.min_word_length]
    
    def process_text(self, text: str) -> List[str]:
        """
        处理文本的完整流程：清洗、分词、去停用词、词干提取或词形还原
        
        Args:
            text: 输入文本
            
        Returns:
            处理后的词列表
        """
        # 清洗文本
        cleaned_text = self.clean_text(text)
        
        # 分词
        tokens = self.tokenize(cleaned_text)
        
        # 过滤短词
        tokens = self.filter_short_words(tokens)
        
        # 移除停用词
        if self.remove_stopwords:
            tokens = self.remove_stopwords_from_tokens(tokens)
        
        # 词干提取
        if self.stemming:
            tokens = self.apply_stemming(tokens)
        
        # 词形还原
        if self.lemmatization:
            tokens = self.apply_lemmatization(tokens)
        
        return tokens
    
    def process_texts(self, texts: List[str]) -> List[List[str]]:
        """
        批量处理多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            处理后的词列表的列表
        """
        return [self.process_text(text) for text in texts]
    
    def get_text_from_tokens(self, tokens: List[str]) -> str:
        """
        将词列表转换回文本
        
        Args:
            tokens: 词列表
            
        Returns:
            合并后的文本
        """
        return " ".join(tokens)


def preprocess_for_sentiment(text: str, advanced: bool = False) -> str:
    """
    专门为情感分析优化的预处理函数
    
    Args:
        text: 输入文本
        advanced: 是否使用高级预处理（spaCy）
        
    Returns:
        预处理后的文本
    """
    if advanced and SPACY_AVAILABLE:
        # 使用spaCy进行高级预处理
        doc = nlp(text)
        # 保留名词、形容词、副词和动词，这些词对情感分析很重要
        tokens = [token.lemma_ for token in doc 
                 if (token.pos_ in ['NOUN', 'ADJ', 'ADV', 'VERB']) 
                 and not token.is_stop]
        return " ".join(tokens)
    else:
        # 使用基本预处理
        preprocessor = TextPreprocessor(remove_stopwords=True, 
                                       stemming=False, 
                                       lemmatization=True)
        tokens = preprocessor.process_text(text)
        return preprocessor.get_text_from_tokens(tokens)


# 辅助函数
def get_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """
    从词列表中生成n-gram
    
    Args:
        tokens: 词列表
        n: n-gram中的n
        
    Returns:
        n-gram列表
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = " ".join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams


if __name__ == "__main__":
    # 示例用法
    sample_text = "This is an example text for preprocessing. It contains some stop words and punctuation!"
    
    preprocessor = TextPreprocessor()
    processed_tokens = preprocessor.process_text(sample_text)
    print(f"Processed tokens: {processed_tokens}")
    
    processed_text = preprocessor.get_text_from_tokens(processed_tokens)
    print(f"Processed text: {processed_text}")
    
    # 高级预处理示例
    advanced_processed = preprocess_for_sentiment(sample_text, advanced=True)
    print(f"Advanced processed text: {advanced_processed}")