# 基于原型网络的少样本NER实现
class PrototypicalNetworkForNER(nn.Module):
    def __init__(self, bert_model_name):
        super(PrototypicalNetworkForNER, self).__init__()
        self.encoder = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, support_inputs, query_inputs):
        # 编码支持集
        support_outputs = self.encoder(**support_inputs)
        support_embeddings = self.dropout(support_outputs.last_hidden_state)
        
        # 编码查询集
        query_outputs = self.encoder(**query_inputs)
        query_embeddings = self.dropout(query_outputs.last_hidden_state)
        
        # 计算类原型 (聚合具有相同标签的token表示)
        # 简化实现，假设support_labels已经处理好
        prototypes = {}  # 每个类别的原型表示
        for idx, label in enumerate(support_labels):
            if label not in prototypes:
                prototypes[label] = []
            prototypes[label].append(support_embeddings[idx])
        
        # 聚合得到类原型
        for label in prototypes:
            prototypes[label] = torch.stack(prototypes[label]).mean(0)
        
        # 计算查询样本与各原型的距离
        distances = {}
        for label, prototype in prototypes.items():
            distance = torch.cdist(query_embeddings, prototype.unsqueeze(0))
            distances[label] = distance
        
        # 返回最近原型的标签
        predictions = []
        for query_idx in range(len(query_inputs)):
            min_distance = float('inf')
            predicted_label = None
            for label, distance in distances.items():
                if distance[query_idx] < min_distance:
                    min_distance = distance[query_idx]
                    predicted_label = label
            predictions.append(predicted_label)
            
        return predictions