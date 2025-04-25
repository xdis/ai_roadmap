from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

# 真实标签和预测标签
true_tags = [
    ['O', 'O', 'B-PER', 'I-PER', 'O'],
    ['B-ORG', 'I-ORG', 'O', 'B-LOC', 'I-LOC']
]
pred_tags = [
    ['O', 'O', 'B-PER', 'I-PER', 'O'],
    ['B-ORG', 'I-ORG', 'O', 'B-PER', 'I-PER']  # LOC预测为PER
]

# 计算指标
p = precision_score(true_tags, pred_tags)
r = recall_score(true_tags, pred_tags)
f1 = f1_score(true_tags, pred_tags)

print(f"Precision: {p:.4f}")
print(f"Recall: {r:.4f}")
print(f"F1-score: {f1:.4f}")

# 详细报告
report = classification_report(true_tags, pred_tags)
print(report)