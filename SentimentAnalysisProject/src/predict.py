"""
模型预测脚本：使用已训练的模型进行情感分析预测
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
import joblib
import json
import logging
from datetime import datetime

# 导入自定义模块
from src.preprocessing import TextPreprocessor
from src.models.classical import SentimentClassifier, EnsembleClassifier
from src.models.deep_learning import PyTorchSentimentModel, TensorFlowSentimentModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_preprocessor(preprocessor_path: str) -> TextPreprocessor:
    """
    加载预处理器
    
    Args:
        preprocessor_path: 预处理器文件路径
        
    Returns:
        加载的预处理器
    """
    logger.info(f"加载预处理器: {preprocessor_path}")
    return joblib.load(preprocessor_path)


def load_feature_extractor(feature_extractor_path: str) -> Any:
    """
    加载特征提取器
    
    Args:
        feature_extractor_path: 特征提取器文件路径
        
    Returns:
        加载的特征提取器
    """
    logger.info(f"加载特征提取器: {feature_extractor_path}")
    return joblib.load(feature_extractor_path)


def load_model(model_path: str, model_type: str) -> Any:
    """
    加载模型
    
    Args:
        model_path: 模型文件路径
        model_type: 模型类型（classical、ensemble、pytorch、tensorflow）
        
    Returns:
        加载的模型
    """
    logger.info(f"加载{model_type}模型: {model_path}")
    
    if model_type == 'classical':
        return SentimentClassifier.load(model_path)
    elif model_type == 'ensemble':
        return EnsembleClassifier.load(model_path)
    elif model_type == 'pytorch':
        return PyTorchSentimentModel.load(model_path)
    elif model_type == 'tensorflow':
        return TensorFlowSentimentModel.load(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def preprocess_text(text: str, preprocessor: TextPreprocessor, advanced: bool = False) -> str:
    """
    预处理文本
    
    Args:
        text: 原始文本
        preprocessor: 预处理器
        advanced: 是否使用高级预处理
        
    Returns:
        预处理后的文本
    """
    if advanced:
        from src.preprocessing import preprocess_for_sentiment
        return preprocess_for_sentiment(text, advanced=True)
    else:
        tokens = preprocessor.process_text(text)
        return preprocessor.get_text_from_tokens(tokens)


def extract_features(text: str, feature_extractor: Any) -> np.ndarray:
    """
    提取特征
    
    Args:
        text: 预处理后的文本
        feature_extractor: 特征提取器
        
    Returns:
        特征向量
    """
    if hasattr(feature_extractor, 'transform'):
        # 单个文本需要包装为列表
        features = feature_extractor.transform([text])
    else:
        # 可能需要其他方式处理
        raise ValueError("特征提取器没有transform方法")
    
    return features


def predict_sentiment(text: str, preprocessor: TextPreprocessor, feature_extractor: Any, 
                     model: Any, return_proba: bool = False, advanced_preprocessing: bool = False) -> Union[str, Tuple[str, Dict[str, float]]]:
    """
    预测文本情感
    
    Args:
        text: 原始文本
        preprocessor: 预处理器
        feature_extractor: 特征提取器
        model: 情感分析模型
        return_proba: 是否返回概率
        advanced_preprocessing: 是否使用高级预处理
        
    Returns:
        预测的情感类别，或情感类别和概率分布
    """
    # 预处理文本
    processed_text = preprocess_text(text, preprocessor, advanced=advanced_preprocessing)
    
    # 提取特征
    features = extract_features(processed_text, feature_extractor)
    
    # 预测
    prediction = model.predict(features)[0]
    
    if return_proba:
        # 获取概率分布
        probas = model.predict_proba(features)[0]
        
        # 将概率与类别对应
        if hasattr(model, 'classes_'):
            classes = model.classes_
        else:
            # 尝试从模型中获取类别信息
            try:
                classes = [key for key in model.class_to_idx.keys()]
            except AttributeError:
                classes = [f"class_{i}" for i in range(len(probas))]
        
        # 创建情感-概率字典
        proba_dict = {cls: float(prob) for cls, prob in zip(classes, probas)}
        
        return prediction, proba_dict
    else:
        return prediction


def batch_predict(texts: List[str], preprocessor: TextPreprocessor, feature_extractor: Any, 
                 model: Any, return_proba: bool = False, advanced_preprocessing: bool = False) -> Union[List[str], Tuple[List[str], List[Dict[str, float]]]]:
    """
    批量预测文本情感
    
    Args:
        texts: 原始文本列表
        preprocessor: 预处理器
        feature_extractor: 特征提取器
        model: 情感分析模型
        return_proba: 是否返回概率
        advanced_preprocessing: 是否使用高级预处理
        
    Returns:
        预测的情感类别列表，或情感类别列表和概率分布列表
    """
    # 预处理文本
    processed_texts = []
    for text in texts:
        processed_text = preprocess_text(text, preprocessor, advanced=advanced_preprocessing)
        processed_texts.append(processed_text)
    
    # 提取特征
    if hasattr(feature_extractor, 'transform'):
        features = feature_extractor.transform(processed_texts)
    else:
        raise ValueError("特征提取器没有transform方法")
    
    # 预测
    predictions = model.predict(features)
    
    if return_proba:
        # 获取概率分布
        probas = model.predict_proba(features)
        
        # 将概率与类别对应
        if hasattr(model, 'classes_'):
            classes = model.classes_
        else:
            # 尝试从模型中获取类别信息
            try:
                classes = [key for key in model.class_to_idx.keys()]
            except AttributeError:
                classes = [f"class_{i}" for i in range(probas.shape[1])]
        
        # 创建情感-概率字典列表
        proba_dicts = []
        for proba in probas:
            proba_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}
            proba_dicts.append(proba_dict)
        
        return predictions.tolist(), proba_dicts
    else:
        return predictions.tolist()


def predict_from_file(input_file: str, output_file: str, preprocessor: TextPreprocessor, 
                     feature_extractor: Any, model: Any,
                     text_column: str = 'text', return_proba: bool = False,
                     advanced_preprocessing: bool = False) -> None:
    """
    从文件读取文本进行批量预测并保存结果
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        preprocessor: 预处理器
        feature_extractor: 特征提取器
        model: 情感分析模型
        text_column: 文本列名
        return_proba: 是否返回概率
        advanced_preprocessing: 是否使用高级预处理
    """
    # 读取文件
    file_ext = os.path.splitext(input_file)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(input_file)
    elif file_ext == '.tsv':
        df = pd.read_csv(input_file, sep='\t')
    elif file_ext == '.json':
        df = pd.read_json(input_file)
    elif file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(input_file)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    # 检查文本列是否存在
    if text_column not in df.columns:
        raise ValueError(f"文本列 '{text_column}' 不在输入文件中")
    
    # 获取文本列表
    texts = df[text_column].tolist()
    
    # 批量预测
    if return_proba:
        predictions, probas = batch_predict(
            texts, preprocessor, feature_extractor, model,
            return_proba=True, advanced_preprocessing=advanced_preprocessing
        )
        
        # 添加预测结果和概率到数据框
        df['predicted_sentiment'] = predictions
        
        # 为每个类别添加概率列
        if len(probas) > 0:
            for cls in probas[0].keys():
                df[f'probability_{cls}'] = [p.get(cls, 0.0) for p in probas]
    else:
        predictions = batch_predict(
            texts, preprocessor, feature_extractor, model,
            return_proba=False, advanced_preprocessing=advanced_preprocessing
        )
        
        # 添加预测结果到数据框
        df['predicted_sentiment'] = predictions
    
    # 保存结果
    file_ext = os.path.splitext(output_file)[1].lower()
    
    if file_ext == '.csv':
        df.to_csv(output_file, index=False)
    elif file_ext == '.tsv':
        df.to_csv(output_file, sep='\t', index=False)
    elif file_ext == '.json':
        df.to_json(output_file, orient='records')
    elif file_ext == '.xlsx' or file_ext == '.xls':
        df.to_excel(output_file, index=False)
    else:
        df.to_csv(output_file, index=False)  # 默认使用CSV
    
    logger.info(f"已处理 {len(texts)} 个文本，结果保存至 {output_file}")


def main(args: argparse.Namespace) -> None:
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    # 加载预处理器
    preprocessor = load_preprocessor(args.preprocessor_path)
    
    # 加载特征提取器
    feature_extractor = load_feature_extractor(args.feature_extractor_path)
    
    # 加载模型
    model = load_model(args.model_path, args.model_type)
    
    # 根据输入方式进行预测
    if args.input_mode == 'interactive':
        # 交互式模式
        logger.info("进入交互式情感分析模式，输入'exit'退出")
        
        while True:
            text = input("\n请输入要分析的文本 (输入'exit'退出): ")
            if text.lower() == 'exit':
                break
            
            if not text.strip():
                continue
            
            try:
                if args.return_proba:
                    prediction, proba_dict = predict_sentiment(
                        text, preprocessor, feature_extractor, model,
                        return_proba=True, advanced_preprocessing=args.advanced_preprocessing
                    )
                    
                    # 根据概率排序情感类别
                    sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
                    
                    print(f"\n预测情感: {prediction}")
                    print("情感概率分布:")
                    for sentiment, prob in sorted_probs:
                        print(f"  - {sentiment}: {prob:.4f}")
                else:
                    prediction = predict_sentiment(
                        text, preprocessor, feature_extractor, model,
                        return_proba=False, advanced_preprocessing=args.advanced_preprocessing
                    )
                    print(f"\n预测情感: {prediction}")
            
            except Exception as e:
                logger.error(f"预测出错: {str(e)}")
                print(f"预测出错: {str(e)}")
    
    elif args.input_mode == 'text':
        # 直接文本模式
        text = args.text
        
        try:
            if args.return_proba:
                prediction, proba_dict = predict_sentiment(
                    text, preprocessor, feature_extractor, model,
                    return_proba=True, advanced_preprocessing=args.advanced_preprocessing
                )
                
                # 根据概率排序情感类别
                sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n预测情感: {prediction}")
                print("情感概率分布:")
                for sentiment, prob in sorted_probs:
                    print(f"  - {sentiment}: {prob:.4f}")
            else:
                prediction = predict_sentiment(
                    text, preprocessor, feature_extractor, model,
                    return_proba=False, advanced_preprocessing=args.advanced_preprocessing
                )
                print(f"\n预测情感: {prediction}")
        
        except Exception as e:
            logger.error(f"预测出错: {str(e)}")
            print(f"预测出错: {str(e)}")
    
    elif args.input_mode == 'file':
        # 文件模式
        try:
            predict_from_file(
                args.input_file, args.output_file,
                preprocessor, feature_extractor, model,
                text_column=args.text_column,
                return_proba=args.return_proba,
                advanced_preprocessing=args.advanced_preprocessing
            )
        except Exception as e:
            logger.error(f"文件处理出错: {str(e)}")
            print(f"文件处理出错: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用已训练的模型进行情感分析预测")
    
    # 模型相关参数
    parser.add_argument("--preprocessor_path", type=str, required=True,
                       help="预处理器文件路径")
    parser.add_argument("--feature_extractor_path", type=str, required=True,
                       help="特征提取器文件路径")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型文件路径")
    parser.add_argument("--model_type", type=str, required=True,
                       choices=['classical', 'ensemble', 'pytorch', 'tensorflow'],
                       help="模型类型")
    
    # 预处理参数
    parser.add_argument("--advanced_preprocessing", action="store_true",
                       help="使用高级预处理技术")
    
    # 预测相关参数
    parser.add_argument("--return_proba", action="store_true",
                       help="是否返回概率分布")
    
    # 输入模式
    input_mode_group = parser.add_mutually_exclusive_group(required=True)
    input_mode_group.add_argument("--interactive", action="store_true",
                                dest="input_mode", default="interactive",
                                help="交互式模式")
    input_mode_group.add_argument("--text", type=str,
                                dest="text",
                                help="要分析的文本")
    input_mode_group.add_argument("--input_file", type=str,
                                dest="input_file",
                                help="输入文件路径")
    
    # 文件模式参数
    parser.add_argument("--output_file", type=str,
                       help="输出文件路径（仅文件模式有效）")
    parser.add_argument("--text_column", type=str, default="text",
                       help="文本列名（仅文件模式有效）")
    
    args = parser.parse_args()
    
    # 设置输入模式
    if args.text:
        args.input_mode = 'text'
    elif args.input_file:
        args.input_mode = 'file'
        if not args.output_file:
            parser.error("使用文件模式时必须提供输出文件路径 (--output_file)")
    else:
        args.input_mode = 'interactive'
    
    main(args)