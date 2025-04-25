class IndustrialNERSystem:
    def __init__(self, models_config, rules_config, knowledge_base_path):
        # 加载模型
        self.models = {}
        for domain, config in models_config.items():
            self.models[domain] = self._load_model(config["model_path"], config["model_type"])
        
        # 加载规则
        self.rules = self._load_rules(rules_config)
        
        # 加载知识库
        self.kb = self._load_knowledge_base(knowledge_base_path)
        
        # 后处理规则
        self.postprocessing = {}
    
    def _load_model(self, model_path, model_type):
        """加载模型"""
        if model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForTokenClassification.from_pretrained(model_path)
            return {"model": model, "tokenizer": tokenizer}
        elif model_type == "bilstm-crf":
            # 加载BiLSTM-CRF模型class IndustrialNERSystem:
    def __init__(self, models_config, rules_config, knowledge_base_path):
        # 加载模型
        self.models = {}
        for domain, config in models_config.items():
            self.models[domain] = self._load_model(config["model_path"], config["model_type"])
        
        # 加载规则
        self.rules = self._load_rules(rules_config)
        
        # 加载知识库
        self.kb = self._load_knowledge_base(knowledge_base_path)
        
        # 后处理规则
        self.postprocessing = {}
    
    def _load_model(self, model_path, model_type):
        """加载模型"""
        if model_type == "bert":
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForTokenClassification.from_pretrained(model_path)
            return {"model": model, "tokenizer": tokenizer}
        elif model_type == "bilstm-crf":
            # 加载BiLSTM-CRF模型