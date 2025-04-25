#!/usr/bin/env python
"""
情感分析系统主脚本 - 集成训练、评估和预测功能
"""
import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_dirs() -> Dict[str, str]:
    """
    创建必要的目录结构
    
    Returns:
        包含路径的字典
    """
    # 定义路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    logs_dir = os.path.join(base_dir, "logs")
    
    # 创建目录
    for directory in [data_dir, models_dir, results_dir, logs_dir]:
        os.makedirs(directory, exist_ok=True)
        
    # 创建子目录
    os.makedirs(os.path.join(models_dir, "classical"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "ensemble"), exist_ok=True)
    os.makedirs(os.path.join(models_dir, "deep_learning"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "evaluation"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "predictions"), exist_ok=True)
    
    # 返回路径字典
    return {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "logs_dir": logs_dir
    }


def train(args: argparse.Namespace) -> None:
    """
    训练情感分析模型
    
    Args:
        args: 命令行参数
    """
    from src.train import train_model
    
    # 确保目录结构
    dirs = setup_dirs()
    
    # 设置模型输出路径
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{args.model_type}_{timestamp}"
    output_dir = os.path.join(dirs["models_dir"], args.model_type, model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置训练参数
    train_params = {
        "data_path": args.data_path,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "model_type": args.model_type,
        "output_dir": output_dir,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_features": args.max_features,
        "ngram_range": (args.min_ngram, args.max_ngram),
        "use_idf": not args.no_idf,
        "max_iter": args.max_iter,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim
    }
    
    # 记录参数
    params_path = os.path.join(output_dir, "params.json")
    with open(params_path, 'w') as f:
        json.dump(train_params, f, indent=4)
    
    logger.info(f"开始训练 {args.model_type} 模型")
    logger.info(f"训练参数: {train_params}")
    
    # 调用训练函数
    train_model(**train_params)
    
    logger.info(f"模型训练完成，保存在 {output_dir}")
    
    # 如果指定了自动评估，则进行评估
    if args.auto_evaluate:
        eval_args = argparse.Namespace(
            data_path=args.data_path,
            text_column=args.text_column,
            label_column=args.label_column,
            model_dir=output_dir,
            output_dir=os.path.join(dirs["results_dir"], "evaluation", model_name)
        )
        evaluate(eval_args)


def evaluate(args: argparse.Namespace) -> None:
    """
    评估情感分析模型
    
    Args:
        args: 命令行参数
    """
    from src.evaluate import main as evaluate_main
    
    # 确保目录结构
    dirs = setup_dirs()
    
    # 确定模型目录
    model_dir = args.model_dir
    
    # 加载模型参数
    params_path = os.path.join(model_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        model_type = model_params.get("model_type", "classical")
    else:
        model_type = "classical"  # 默认模型类型
        logger.warning(f"找不到模型参数文件: {params_path}，使用默认模型类型: {model_type}")
    
    # 确定评估输出目录
    if args.output_dir:
        output_dir = args.output_dir
    else:
        model_name = os.path.basename(model_dir)
        output_dir = os.path.join(dirs["results_dir"], "evaluation", model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 组装评估参数
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    feature_extractor_path = os.path.join(model_dir, "feature_extractor.joblib")
    model_path = os.path.join(model_dir, "model.joblib")
    
    # 构建评估参数
    eval_args = argparse.Namespace(
        data_path=args.data_path,
        text_column=args.text_column,
        label_column=args.label_column,
        preprocessor_path=preprocessor_path,
        feature_extractor_path=feature_extractor_path,
        model_path=model_path,
        model_type=model_type,
        advanced_preprocessing=args.advanced_preprocessing,
        output_dir=output_dir
    )
    
    logger.info(f"开始评估模型: {model_dir}")
    evaluate_main(eval_args)
    logger.info(f"模型评估完成，结果保存在 {output_dir}")


def predict(args: argparse.Namespace) -> None:
    """
    使用模型进行情感预测
    
    Args:
        args: 命令行参数
    """
    from src.predict import predict_sentiment
    
    # 确保目录结构
    dirs = setup_dirs()
    
    # 确定模型目录
    model_dir = args.model_dir
    
    # 加载模型参数
    params_path = os.path.join(model_dir, "params.json")
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            model_params = json.load(f)
        model_type = model_params.get("model_type", "classical")
    else:
        model_type = "classical"  # 默认模型类型
        logger.warning(f"找不到模型参数文件: {params_path}，使用默认模型类型: {model_type}")
    
    # 确定预测输出目录
    if args.output_path:
        output_path = args.output_path
        output_dir = os.path.dirname(output_path)
    else:
        model_name = os.path.basename(model_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(dirs["results_dir"], "predictions", model_name)
        output_path = os.path.join(output_dir, f"predictions_{timestamp}.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 组装预测参数
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    feature_extractor_path = os.path.join(model_dir, "feature_extractor.joblib")
    model_path = os.path.join(model_dir, "model.joblib")
    
    # 预测数据源
    if args.text:
        # 单个文本预测
        texts = [args.text]
        is_file = False
    else:
        # 从文件预测
        is_file = True
        # 检查文件是否存在
        if not os.path.exists(args.input_path):
            logger.error(f"输入文件不存在: {args.input_path}")
            return
    
    logger.info(f"开始进行情感预测，使用模型: {model_dir}")
    
    if is_file:
        # 文件预测
        predict_sentiment(
            input_path=args.input_path,
            output_path=output_path,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            feature_extractor_path=feature_extractor_path,
            model_type=model_type,
            text_column=args.text_column,
            include_probabilities=args.include_probabilities,
            batch_size=args.batch_size
        )
        logger.info(f"文件预测完成，结果保存在 {output_path}")
    else:
        # 单个文本预测
        result = predict_sentiment(
            texts=texts,
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            feature_extractor_path=feature_extractor_path,
            model_type=model_type,
            include_probabilities=args.include_probabilities
        )
        
        # 打印结果
        for text, pred in zip(texts, result):
            if args.include_probabilities and isinstance(pred, tuple):
                label, probs = pred
                print(f"文本: {text}")
                print(f"预测标签: {label}")
                print(f"预测概率: {probs}")
            else:
                print(f"文本: {text}")
                print(f"预测标签: {pred}")
        
        # 如果指定了输出路径，也保存结果
        if args.output_path:
            if args.include_probabilities and isinstance(result[0], tuple):
                labels, probs_list = zip(*result)
                df = pd.DataFrame({
                    'text': texts,
                    'predicted_label': labels
                })
                # 添加概率列
                if isinstance(probs_list[0], dict):
                    for label in probs_list[0].keys():
                        df[f'prob_{label}'] = [p.get(label, 0) for p in probs_list]
                elif isinstance(probs_list[0], (list, np.ndarray)):
                    for i, prob in enumerate(probs_list[0]):
                        df[f'prob_{i}'] = [p[i] for p in probs_list]
            else:
                df = pd.DataFrame({
                    'text': texts,
                    'predicted_label': result
                })
            
            df.to_csv(output_path, index=False)
            logger.info(f"单个文本预测结果也保存在 {output_path}")


def find_latest_model(model_type: str = "classical") -> Optional[str]:
    """
    查找最新训练的模型
    
    Args:
        model_type: 模型类型
        
    Returns:
        最新模型的路径
    """
    dirs = setup_dirs()
    models_type_dir = os.path.join(dirs["models_dir"], model_type)
    
    if not os.path.exists(models_type_dir):
        logger.warning(f"模型目录不存在: {models_type_dir}")
        return None
    
    # 获取所有模型目录
    model_dirs = [os.path.join(models_type_dir, d) for d in os.listdir(models_type_dir)
                  if os.path.isdir(os.path.join(models_type_dir, d))]
    
    if not model_dirs:
        logger.warning(f"没有找到 {model_type} 类型的模型")
        return None
    
    # 按修改时间排序
    model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return model_dirs[0]


def list_models(args: argparse.Namespace) -> None:
    """
    列出所有可用的模型
    
    Args:
        args: 命令行参数
    """
    dirs = setup_dirs()
    models_dir = dirs["models_dir"]
    
    # 列出所有模型类型
    model_types = [d for d in os.listdir(models_dir) 
                  if os.path.isdir(os.path.join(models_dir, d))]
    
    if not model_types:
        print("未找到任何模型")
        return
    
    print("可用的情感分析模型:")
    print("=" * 80)
    
    for model_type in model_types:
        model_type_dir = os.path.join(models_dir, model_type)
        model_names = [d for d in os.listdir(model_type_dir)
                       if os.path.isdir(os.path.join(model_type_dir, d))]
        
        if not model_names:
            continue
        
        print(f"\n{model_type.upper()} 模型:")
        print("-" * 80)
        
        # 按修改时间排序
        model_paths = [os.path.join(model_type_dir, name) for name in model_names]
        model_info = [(path, os.path.getmtime(path)) for path in model_paths]
        model_info.sort(key=lambda x: x[1], reverse=True)
        
        for i, (path, mtime) in enumerate(model_info, 1):
            name = os.path.basename(path)
            timestamp = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            
            # 尝试加载模型参数
            params_path = os.path.join(path, "params.json")
            params_info = ""
            if os.path.exists(params_path):
                try:
                    with open(params_path, 'r') as f:
                        params = json.load(f)
                    # 提取一些关键参数显示
                    features = params.get("max_features", "未知")
                    ngram = params.get("ngram_range", ["未知", "未知"])
                    params_info = f"特征数: {features}, N-gram: {ngram[0]}-{ngram[1]}"
                except:
                    params_info = "无法加载参数"
            
            print(f"{i}. {name}")
            print(f"   路径: {path}")
            print(f"   创建时间: {timestamp}")
            if params_info:
                print(f"   参数: {params_info}")
            
            # 如果详细模式，显示更多信息
            if args.verbose:
                # 检查模型文件
                files = os.listdir(path)
                print(f"   文件: {', '.join(files)}")
                
                # 如果有评估结果，显示主要指标
                eval_dir = os.path.join(dirs["results_dir"], "evaluation", name)
                if os.path.exists(eval_dir):
                    results_path = os.path.join(eval_dir, "evaluation_results.json")
                    if os.path.exists(results_path):
                        try:
                            with open(results_path, 'r') as f:
                                results = json.load(f)
                            accuracy = results.get("accuracy", "未知")
                            f1 = results.get("f1_weighted", "未知")
                            print(f"   性能: 准确率={accuracy:.4f}, F1分数={f1:.4f}")
                        except:
                            print("   评估结果无法加载")
                    else:
                        print("   未找到评估结果")


def main():
    """主函数"""
    # 创建主解析器
    parser = argparse.ArgumentParser(
        description="情感分析系统 - 训练、评估和预测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练子命令
    train_parser = subparsers.add_parser("train", help="训练情感分析模型")
    train_parser.add_argument("--data_path", type=str, required=True,
                            help="训练数据文件路径")
    train_parser.add_argument("--text_column", type=str, default="text",
                            help="文本列名")
    train_parser.add_argument("--label_column", type=str, default="label",
                            help="标签列名")
    train_parser.add_argument("--model_type", type=str, default="classical",
                            choices=["classical", "ensemble", "pytorch", "tensorflow"],
                            help="模型类型")
    train_parser.add_argument("--test_size", type=float, default=0.2,
                            help="测试集比例")
    train_parser.add_argument("--random_state", type=int, default=42,
                            help="随机种子")
    train_parser.add_argument("--max_features", type=int, default=10000,
                            help="最大特征数")
    train_parser.add_argument("--min_ngram", type=int, default=1,
                            help="N-gram 最小值")
    train_parser.add_argument("--max_ngram", type=int, default=3,
                            help="N-gram 最大值")
    train_parser.add_argument("--no_idf", action="store_true",
                            help="不使用 IDF 权重")
    train_parser.add_argument("--max_iter", type=int, default=1000,
                            help="最大迭代次数")
    train_parser.add_argument("--batch_size", type=int, default=64,
                            help="批次大小")
    train_parser.add_argument("--learning_rate", type=float, default=0.001,
                            help="学习率")
    train_parser.add_argument("--embedding_dim", type=int, default=100,
                            help="词嵌入维度")
    train_parser.add_argument("--hidden_dim", type=int, default=128,
                            help="隐藏层维度")
    train_parser.add_argument("--auto_evaluate", action="store_true",
                            help="训练后自动评估模型")
    
    # 评估子命令
    eval_parser = subparsers.add_parser("evaluate", help="评估情感分析模型")
    eval_parser.add_argument("--data_path", type=str, required=True,
                           help="测试数据文件路径")
    eval_parser.add_argument("--text_column", type=str, default="text",
                           help="文本列名")
    eval_parser.add_argument("--label_column", type=str, default="label",
                           help="标签列名")
    eval_parser.add_argument("--model_dir", type=str, 
                           help="模型目录路径，如果不提供将使用最新的模型")
    eval_parser.add_argument("--output_dir", type=str,
                           help="评估结果输出目录")
    eval_parser.add_argument("--advanced_preprocessing", action="store_true",
                           help="使用高级预处理")
    
    # 预测子命令
    predict_parser = subparsers.add_parser("predict", help="使用模型进行情感预测")
    predict_input_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_input_group.add_argument("--text", type=str,
                                    help="要预测的单个文本")
    predict_input_group.add_argument("--input_path", type=str,
                                    help="要预测的文件路径")
    predict_parser.add_argument("--output_path", type=str,
                              help="预测结果输出路径")
    predict_parser.add_argument("--model_dir", type=str,
                              help="模型目录路径，如果不提供将使用最新的模型")
    predict_parser.add_argument("--text_column", type=str, default="text",
                              help="输入文件中的文本列名")
    predict_parser.add_argument("--include_probabilities", action="store_true",
                              help="包含预测概率")
    predict_parser.add_argument("--batch_size", type=int, default=64,
                              help="批处理大小")
    
    # 列出模型子命令
    list_parser = subparsers.add_parser("list", help="列出可用的模型")
    list_parser.add_argument("--verbose", "-v", action="store_true",
                           help="显示详细信息")
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有提供命令，打印帮助并退出
    if not args.command:
        parser.print_help()
        return
    
    # 对于评估和预测命令，如果没有提供模型目录，使用最新的模型
    if args.command in ["evaluate", "predict"] and not hasattr(args, "model_dir"):
        model_type = "classical"  # 默认使用传统模型
        latest_model = find_latest_model(model_type)
        if not latest_model:
            logger.error(f"未找到 {model_type} 类型的模型")
            return
        args.model_dir = latest_model
        logger.info(f"使用最新的模型: {latest_model}")
    
    # 执行对应的命令
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "list":
        list_models(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.exception(f"发生错误: {e}")
        sys.exit(1)