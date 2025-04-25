"""
模型评估脚本：评估情感分析模型的性能
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import joblib
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
)

# 导入自定义模块
from src.preprocessing import TextPreprocessor
from src.models.classical import SentimentClassifier, EnsembleClassifier
from src.models.deep_learning import PyTorchSentimentModel, TensorFlowSentimentModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(data_path: str, text_column: str = 'text', label_column: str = 'label',
             encoding: str = 'utf-8', sep: str = ',') -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    加载数据集
    
    Args:
        data_path: 数据文件路径
        text_column: 文本列名
        label_column: 标签列名
        encoding: 文件编码
        sep: 分隔符（对于CSV文件）
        
    Returns:
        数据框、文本列表和标签列表
    """
    # 检查文件扩展名
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.csv':
        df = pd.read_csv(data_path, encoding=encoding, sep=sep)
    elif file_ext == '.tsv':
        df = pd.read_csv(data_path, encoding=encoding, sep='\t')
    elif file_ext == '.json':
        df = pd.read_json(data_path, encoding=encoding)
    elif file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # 检查必要的列是否存在
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in the data")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in the data")
    
    # 提取文本和标签
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    logger.info(f"Loaded {len(texts)} samples from {data_path}")
    return df, texts, labels


def load_model_pipeline(preprocessor_path: str, feature_extractor_path: str,
                      model_path: str, model_type: str) -> Tuple[Any, Any, Any]:
    """
    加载模型管道（预处理器、特征提取器和模型）
    
    Args:
        preprocessor_path: 预处理器文件路径
        feature_extractor_path: 特征提取器文件路径
        model_path: 模型文件路径
        model_type: 模型类型
        
    Returns:
        预处理器、特征提取器和模型
    """
    # 加载预处理器
    logger.info(f"加载预处理器: {preprocessor_path}")
    preprocessor = joblib.load(preprocessor_path)
    
    # 加载特征提取器
    logger.info(f"加载特征提取器: {feature_extractor_path}")
    feature_extractor = joblib.load(feature_extractor_path)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    if model_type == 'classical':
        model = SentimentClassifier.load(model_path)
    elif model_type == 'ensemble':
        model = EnsembleClassifier.load(model_path)
    elif model_type == 'pytorch':
        model = PyTorchSentimentModel.load(model_path)
    elif model_type == 'tensorflow':
        model = TensorFlowSentimentModel.load(model_path)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return preprocessor, feature_extractor, model


def preprocess_and_extract_features(texts: List[str], preprocessor: Any, 
                                   feature_extractor: Any, 
                                   advanced_preprocessing: bool = False) -> np.ndarray:
    """
    预处理文本并提取特征
    
    Args:
        texts: 原始文本列表
        preprocessor: 预处理器
        feature_extractor: 特征提取器
        advanced_preprocessing: 是否使用高级预处理
        
    Returns:
        特征矩阵
    """
    # 预处理文本
    logger.info("预处理文本...")
    processed_texts = []
    
    if advanced_preprocessing:
        from src.preprocessing import preprocess_for_sentiment
        for text in texts:
            processed_text = preprocess_for_sentiment(text, advanced=True)
            processed_texts.append(processed_text)
    else:
        for text in texts:
            tokens = preprocessor.process_text(text)
            processed_text = preprocessor.get_text_from_tokens(tokens)
            processed_texts.append(processed_text)
    
    # 提取特征
    logger.info("提取特征...")
    features = feature_extractor.transform(processed_texts)
    
    return features


def evaluate_model(model: Any, X_test: np.ndarray, y_test: List[str], 
                 class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        class_names: 类别名称列表
        
    Returns:
        评估结果
    """
    logger.info("开始评估模型...")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 获取预测概率（如果可用）
    try:
        y_proba = model.predict_proba(X_test)
        has_probas = True
    except (AttributeError, NotImplementedError):
        y_proba = None
        has_probas = False
    
    # 获取类别名称
    if class_names is None:
        if hasattr(model, 'classes_'):
            class_names = model.classes_
        else:
            # 尝试从预测和实际标签中提取唯一类别
            class_names = sorted(list(set(y_test).union(set(y_pred))))
    
    # 计算基本指标
    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # 计算每个类别的指标
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    
    # 将结果组织成字典
    results = {
        'accuracy': accuracy,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'class_report': class_report,
        'confusion_matrix': cm.tolist(),
        'true_labels': y_test,
        'predicted_labels': y_pred.tolist() if isinstance(y_pred, np.ndarray) else y_pred,
        'class_names': class_names
    }
    
    # 如果有概率预测，添加ROC AUC和PR AUC
    if has_probas and len(class_names) > 1:
        binary_classification = len(class_names) == 2
        
        if binary_classification:
            # 二分类情况
            positive_idx = 1  # 假设正类是索引1
            y_test_bin = [1 if label == class_names[positive_idx] else 0 for label in y_test]
            y_score = y_proba[:, positive_idx]
            
            # ROC曲线
            fpr, tpr, _ = roc_curve(y_test_bin, y_score)
            roc_auc = auc(fpr, tpr)
            results['binary_roc'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc}
            
            # PR曲线
            precision, recall, _ = precision_recall_curve(y_test_bin, y_score)
            pr_auc = auc(recall, precision)
            results['binary_pr'] = {'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': pr_auc}
        else:
            # 多分类情况
            results['multiclass_roc'] = {}
            results['multiclass_pr'] = {}
            
            # 为每个类别计算
            for i, class_name in enumerate(class_names):
                # 将问题转化为二分类
                y_test_bin = [1 if label == class_name else 0 for label in y_test]
                try:
                    y_score = y_proba[:, i]
                    
                    # ROC曲线
                    fpr, tpr, _ = roc_curve(y_test_bin, y_score)
                    roc_auc = auc(fpr, tpr)
                    results['multiclass_roc'][class_name] = {
                        'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc
                    }
                    
                    # PR曲线
                    precision, recall, _ = precision_recall_curve(y_test_bin, y_score)
                    pr_auc = auc(recall, precision)
                    results['multiclass_pr'][class_name] = {
                        'precision': precision.tolist(), 'recall': recall.tolist(), 'auc': pr_auc
                    }
                except:
                    logger.warning(f"无法为类别 {class_name} 计算ROC或PR曲线")
    
    # 输出主要评估指标
    logger.info(f"准确率: {accuracy:.4f}")
    logger.info(f"加权精确率: {precision_weighted:.4f}")
    logger.info(f"加权召回率: {recall_weighted:.4f}")
    logger.info(f"加权F1分数: {f1_weighted:.4f}")
    
    # 如果是二分类且有概率，输出AUC
    if has_probas and len(class_names) == 2:
        logger.info(f"ROC AUC: {results['binary_roc']['auc']:.4f}")
        logger.info(f"PR AUC: {results['binary_pr']['auc']:.4f}")
    
    return results


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names: List[str], 
                         output_path: str, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制混淆矩阵
    
    Args:
        confusion_matrix: 混淆矩阵
        class_names: 类别名称
        output_path: 输出文件路径
        figsize: 图形大小
    """
    plt.figure(figsize=figsize)
    
    # 如果类别太多，调整字体大小
    fontsize = 10 if len(class_names) <= 10 else 8
    
    # 绘制混淆矩阵热图
    ax = sns.heatmap(
        confusion_matrix, 
        annot=True, 
        cmap="Blues", 
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.ylabel('True Label', fontsize=fontsize + 2)
    plt.xlabel('Predicted Label', fontsize=fontsize + 2)
    plt.title('Confusion Matrix', fontsize=fontsize + 4)
    
    # 调整刻度标签字体大小
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=fontsize)
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"混淆矩阵已保存至 {output_path}")


def plot_roc_curves(results: Dict[str, Any], output_path: str, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制ROC曲线
    
    Args:
        results: 模型评估结果
        output_path: 输出文件路径
        figsize: 图形大小
    """
    if 'binary_roc' in results:
        # 二分类ROC曲线
        plt.figure(figsize=figsize)
        
        fpr = results['binary_roc']['fpr']
        tpr = results['binary_roc']['tpr']
        roc_auc = results['binary_roc']['auc']
        
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"ROC曲线已保存至 {output_path}")
        
    elif 'multiclass_roc' in results:
        # 多分类ROC曲线
        plt.figure(figsize=figsize)
        
        for class_name, roc_data in results['multiclass_roc'].items():
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            roc_auc = roc_data['auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{class_name} (area = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (One-vs-Rest)')
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"多分类ROC曲线已保存至 {output_path}")


def plot_pr_curves(results: Dict[str, Any], output_path: str, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    绘制精确率-召回率曲线
    
    Args:
        results: 模型评估结果
        output_path: 输出文件路径
        figsize: 图形大小
    """
    if 'binary_pr' in results:
        # 二分类PR曲线
        plt.figure(figsize=figsize)
        
        precision = results['binary_pr']['precision']
        recall = results['binary_pr']['recall']
        pr_auc = results['binary_pr']['auc']
        
        plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"PR曲线已保存至 {output_path}")
        
    elif 'multiclass_pr' in results:
        # 多分类PR曲线
        plt.figure(figsize=figsize)
        
        for class_name, pr_data in results['multiclass_pr'].items():
            precision = pr_data['precision']
            recall = pr_data['recall']
            pr_auc = pr_data['auc']
            
            plt.plot(recall, precision, lw=2, label=f'{class_name} (area = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (One-vs-Rest)')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"多分类PR曲线已保存至 {output_path}")


def plot_metrics_by_class(results: Dict[str, Any], output_path: str, figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    绘制每个类别的评估指标柱状图
    
    Args:
        results: 模型评估结果
        output_path: 输出文件路径
        figsize: 图形大小
    """
    class_report = results['class_report']
    class_names = [name for name in class_report.keys() if name not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # 提取每个类别的精确率、召回率和F1分数
    precisions = [class_report[name]['precision'] for name in class_names]
    recalls = [class_report[name]['recall'] for name in class_names]
    f1_scores = [class_report[name]['f1-score'] for name in class_names]
    
    # 设置图形
    plt.figure(figsize=figsize)
    
    # 设置条形的位置
    x = np.arange(len(class_names))
    width = 0.25
    
    # 绘制条形图
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1_scores, width, label='F1-Score')
    
    plt.ylabel('Score')
    plt.title('Metrics by Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    
    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 显示每个条形的值
    for i, v in enumerate(precisions):
        plt.text(i - width, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(recalls):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(f1_scores):
        plt.text(i + width, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    # 设置y轴范围
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"类别指标图已保存至 {output_path}")


def analyze_errors(results: Dict[str, Any], texts: List[str], output_path: str) -> None:
    """
    分析模型错误预测
    
    Args:
        results: 模型评估结果
        texts: 原始文本列表
        output_path: 输出文件路径
    """
    true_labels = results['true_labels']
    predicted_labels = results['predicted_labels']
    class_names = results['class_names']
    
    # 创建错误分析数据框
    error_df = pd.DataFrame({
        'text': texts,
        'true_label': true_labels,
        'predicted_label': predicted_labels
    })
    
    # 筛选出错误预测
    error_df = error_df[error_df['true_label'] != error_df['predicted_label']]
    
    # 添加是否为错误预测的列
    error_df['is_error'] = True
    
    # 按类别统计错误
    error_counts = {}
    for true_cls in class_names:
        error_counts[true_cls] = {}
        for pred_cls in class_names:
            if true_cls != pred_cls:
                count = len(error_df[(error_df['true_label'] == true_cls) & 
                                     (error_df['predicted_label'] == pred_cls)])
                error_counts[true_cls][pred_cls] = count
    
    # 保存错误分析
    with open(output_path, 'w') as f:
        f.write("# 错误分析报告\n\n")
        
        # 总体错误统计
        f.write("## 总体错误统计\n\n")
        f.write(f"- 总样本数: {len(texts)}\n")
        f.write(f"- 正确预测数: {len(texts) - len(error_df)}\n")
        f.write(f"- 错误预测数: {len(error_df)}\n")
        f.write(f"- 错误率: {len(error_df) / len(texts) * 100:.2f}%\n\n")
        
        # 按类别错误统计
        f.write("## 按类别错误统计\n\n")
        f.write("| 真实类别 | 错误预测为 | 数量 |\n")
        f.write("|---------|----------|------|\n")
        
        for true_cls in class_names:
            for pred_cls, count in error_counts[true_cls].items():
                if count > 0:
                    f.write(f"| {true_cls} | {pred_cls} | {count} |\n")
        
        f.write("\n")
        
        # 错误样本展示
        f.write("## 错误样本展示\n\n")
        
        # 限制最多展示100个错误样本
        max_errors_to_show = min(100, len(error_df))
        for i, (_, row) in enumerate(error_df.head(max_errors_to_show).iterrows()):
            f.write(f"### 错误样本 {i+1}\n\n")
            f.write(f"- 文本: {row['text']}\n")
            f.write(f"- 真实标签: {row['true_label']}\n")
            f.write(f"- 预测标签: {row['predicted_label']}\n\n")
    
    logger.info(f"错误分析报告已保存至 {output_path}")
    
    # 如果错误数量太多，还可以保存完整的错误数据框
    if len(error_df) > max_errors_to_show:
        csv_path = os.path.splitext(output_path)[0] + "_full.csv"
        error_df.to_csv(csv_path, index=False)
        logger.info(f"完整错误数据已保存至 {csv_path}")


def create_evaluation_report(results: Dict[str, Any], output_path: str) -> None:
    """
    创建评估报告
    
    Args:
        results: 模型评估结果
        output_path: 输出文件路径
    """
    # 提取结果
    accuracy = results['accuracy']
    precision_weighted = results['precision_weighted']
    recall_weighted = results['recall_weighted']
    f1_weighted = results['f1_weighted']
    class_report = results['class_report']
    class_names = results['class_names']
    
    # 创建报告
    with open(output_path, 'w') as f:
        f.write("# 情感分析模型评估报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体性能
        f.write("## 总体性能指标\n\n")
        f.write(f"- 准确率: {accuracy:.4f}\n")
        f.write(f"- 加权精确率: {precision_weighted:.4f}\n")
        f.write(f"- 加权召回率: {recall_weighted:.4f}\n")
        f.write(f"- 加权F1分数: {f1_weighted:.4f}\n\n")
        
        # 如果有AUC指标，添加到报告中
        if 'binary_roc' in results:
            f.write(f"- ROC AUC: {results['binary_roc']['auc']:.4f}\n")
            f.write(f"- PR AUC: {results['binary_pr']['auc']:.4f}\n\n")
        
        # 各类别性能
        f.write("## 各类别性能指标\n\n")
        f.write("| 类别 | 精确率 | 召回率 | F1分数 | 支持度 |\n")
        f.write("|------|-------|-------|-------|-------|\n")
        
        for cls in class_names:
            cls_data = class_report[cls]
            f.write(f"| {cls} | {cls_data['precision']:.4f} | {cls_data['recall']:.4f} | "
                   f"{cls_data['f1-score']:.4f} | {cls_data['support']} |\n")
        
        # 添加宏平均和加权平均
        macro_avg = class_report['macro avg']
        weighted_avg = class_report['weighted avg']
        
        f.write(f"| 宏平均 | {macro_avg['precision']:.4f} | {macro_avg['recall']:.4f} | "
               f"{macro_avg['f1-score']:.4f} | {macro_avg['support']} |\n")
        f.write(f"| 加权平均 | {weighted_avg['precision']:.4f} | {weighted_avg['recall']:.4f} | "
               f"{weighted_avg['f1-score']:.4f} | {weighted_avg['support']} |\n\n")
        
        # 引用图表
        f.write("## 可视化评估\n\n")
        f.write("### 混淆矩阵\n\n")
        f.write("![混淆矩阵](./confusion_matrix.png)\n\n")
        
        f.write("### 类别指标图\n\n")
        f.write("![类别指标图](./class_metrics.png)\n\n")
        
        if 'binary_roc' in results or 'multiclass_roc' in results:
            f.write("### ROC曲线\n\n")
            f.write("![ROC曲线](./roc_curve.png)\n\n")
        
        if 'binary_pr' in results or 'multiclass_pr' in results:
            f.write("### 精确率-召回率曲线\n\n")
            f.write("![PR曲线](./pr_curve.png)\n\n")
        
        # 总结
        f.write("## 评估总结\n\n")
        
        # 找出表现最好和最差的类别
        best_class = max(class_names, key=lambda cls: class_report[cls]['f1-score'])
        worst_class = min(class_names, key=lambda cls: class_report[cls]['f1-score'])
        
        f.write(f"- 表现最好的类别: {best_class}，F1分数: {class_report[best_class]['f1-score']:.4f}\n")
        f.write(f"- 表现最差的类别: {worst_class}，F1分数: {class_report[worst_class]['f1-score']:.4f}\n\n")
        
        # 建议
        f.write("## 改进建议\n\n")
        f.write("1. 针对表现较差的类别增加训练样本\n")
        f.write("2. 考虑使用更高级的特征提取方法\n")
        f.write("3. 尝试使用不同的模型或模型组合\n")
        f.write("4. 优化模型超参数\n")
        f.write("5. 改进文本预处理步骤\n")
    
    logger.info(f"评估报告已保存至 {output_path}")


def main(args: argparse.Namespace) -> None:
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    df, texts, labels = load_data(
        args.data_path,
        text_column=args.text_column,
        label_column=args.label_column
    )
    
    # 加载模型管道
    preprocessor, feature_extractor, model = load_model_pipeline(
        args.preprocessor_path,
        args.feature_extractor_path,
        args.model_path,
        args.model_type
    )
    
    # 预处理和特征提取
    features = preprocess_and_extract_features(
        texts,
        preprocessor,
        feature_extractor,
        advanced_preprocessing=args.advanced_preprocessing
    )
    
    # 评估模型
    results = evaluate_model(model, features, labels)
    
    # 绘制混淆矩阵
    cm = np.array(results['confusion_matrix'])
    cm_output_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, results['class_names'], cm_output_path)
    
    # 绘制类别指标图
    metrics_output_path = os.path.join(args.output_dir, "class_metrics.png")
    plot_metrics_by_class(results, metrics_output_path)
    
    # 如果有ROC数据，绘制ROC曲线
    if 'binary_roc' in results or 'multiclass_roc' in results:
        roc_output_path = os.path.join(args.output_dir, "roc_curve.png")
        plot_roc_curves(results, roc_output_path)
    
    # 如果有PR数据，绘制PR曲线
    if 'binary_pr' in results or 'multiclass_pr' in results:
        pr_output_path = os.path.join(args.output_dir, "pr_curve.png")
        plot_pr_curves(results, pr_output_path)
    
    # 进行错误分析
    error_analysis_path = os.path.join(args.output_dir, "error_analysis.md")
    analyze_errors(results, texts, error_analysis_path)
    
    # 创建评估报告
    report_path = os.path.join(args.output_dir, "evaluation_report.md")
    create_evaluation_report(results, report_path)
    
    # 保存完整评估结果
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    
    # 移除不便于JSON序列化的内容
    results_for_json = {k: v for k, v in results.items() 
                        if k not in ['true_labels', 'predicted_labels']}
    
    with open(results_path, 'w') as f:
        json.dump(results_for_json, f, indent=4)
    
    logger.info(f"完整评估结果已保存至 {results_path}")
    logger.info(f"评估结束，所有输出都保存在 {args.output_dir} 目录下")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估情感分析模型的性能")
    
    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                       help="测试数据文件路径")
    parser.add_argument("--text_column", type=str, default="text",
                       help="文本列名")
    parser.add_argument("--label_column", type=str, default="label",
                       help="标签列名")
    
    # 模型参数
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
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./evaluation",
                       help="评估结果输出目录")
    
    args = parser.parse_args()
    
    main(args)