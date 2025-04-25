class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_vocab_size=None, char_embedding_dim=None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        
        # 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # 可选：字符级特征
        self.use_char = char_vocab_size is not None
        if self.use_char:
            self.char_embeds = nn.Embedding(char_vocab_size, char_embedding_dim)
            self.char_lstm = nn.LSTM(char_embedding_dim, char_embedding_dim//2, 
                                     bidirectional=True, batch_first=True)
            lstm_input_dim = embedding_dim + char_embedding_dim
        else:
            lstm_input_dim = embedding_dim
        
        # BiLSTM层
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim // 2,
                           num_layers=1, bidirectional=True, batch_first=True)
        
        # 线性层映射到标签空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        
        # CRF层的转移矩阵
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        
        # 转移约束：句首不能是I-X，句尾不能是B-X
        self.transitions.data[tag_to_ix['START'], :] = -10000
        for tag in tag_to_ix:
            if tag.startswith('I-'):
                self.transitions.data[:, tag_to_ix[tag]] = -10000
                self.transitions.data[tag_to_ix['START'], tag_to_ix[tag]] = -10000
            
    def _forward_alg(self, feats):
        """前向算法计算归一化因子"""
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix['START']] = 0.
        
        forward_var = init_alphas
        
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix['STOP']]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence, chars=None):
        """从BiLSTM获取发射分数"""
        embeds = self.word_embeds(sentence)
        
        if self.use_char and chars is not None:
            batch_size, seq_len, char_len = chars.size()
            chars = chars.view(batch_size * seq_len, char_len)
            char_embeds = self.char_embeds(chars)
            char_lstm_out, _ = self.char_lstm(char_embeds)
            char_lstm_out = char_lstm_out[:, -1, :]  # 取最后一个状态
            char_lstm_out = char_lstm_out.view(batch_size, seq_len, -1)
            embeds = torch.cat([embeds, char_lstm_out], dim=2)
            
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        """计算给定标签序列的分数"""
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix['START']], dtype=torch.long), tags])
        
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix['STOP'], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        """使用Viterbi算法解码最佳路径"""
        backpointers = []
        
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix['START']] = 0
        
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        
        terminal_var = forward_var + self.transitions[self.tag_to_ix['STOP']]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]
        
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        
        best_path.reverse()
        return path_score, best_path
    
    def forward(self, sentence, chars=None, tags=None):
        """模型前向传播"""
        lstm_feats = self._get_lstm_features(sentence, chars)
        
        if tags is not None:
            # 训练模式：计算CRF损失
            forward_score = self._forward_alg(lstm_feats)
            gold_score = self._score_sentence(lstm_feats, tags)
            return forward_score - gold_score  # 负对数似然
        else:
            # 预测模式：Viterbi解码
            score, tag_seq = self._viterbi_decode(lstm_feats)
            return score, tag_seq