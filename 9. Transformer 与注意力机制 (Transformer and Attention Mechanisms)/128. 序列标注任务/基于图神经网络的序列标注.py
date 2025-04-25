import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GraphNeuralNER(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
        
        # 图卷积层
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        
        # 输出层
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)  # 结合LSTM和GCN特征
        
    def forward(self, x, edge_index, batch_idx):
        # LSTM特征提取
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        
        # 图卷积特征提取
        gcn_out = F.relu(self.gcn1(lstm_out, edge_index))
        gcn_out = self.gcn2(gcn_out, edge_index)
        
        # 特征融合
        combined_features = torch.cat([lstm_out, gcn_out], dim=-1)
        
        # 分类
        logits = self.classifier(combined_features)
        return logits