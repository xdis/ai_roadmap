# 使用sklearn-crfsuite的CRF实现示例
import sklearn_crfsuite

def word2features(sent, i):
    word = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    # 添加前后词特征...
    return features

# 提取特征
X_train = [[word2features(s, i) for i in range(len(s))] for s in train_data]
y_train = [[label for _, label in s] for s in train_data]

# 训练CRF模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)