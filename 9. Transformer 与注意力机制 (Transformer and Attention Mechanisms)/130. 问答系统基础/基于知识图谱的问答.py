# 简化的知识图谱问答示意
class KnowledgeGraphQA:
    def __init__(self):
        # 示例知识图谱(简化为三元组)
        self.kg_triples = [
            ("爱因斯坦", "出生于", "德国乌尔姆"),
            ("爱因斯坦", "出生日期", "1879年3月14日"),
            ("爱因斯坦", "职业", "物理学家"),
            ("爱因斯坦", "获得", "诺贝尔物理学奖"),
            ("诺贝尔物理学奖", "颁发年份", "1921年"),
            ("特殊相对论", "提出者", "爱因斯坦"),
            ("特殊相对论", "提出年份", "1905年")
        ]
        
        # NER和关系抽取模型(实际应用中需要专门训练)
        self.ner_model = None  # 实际中需要加载
        self.relation_extractor = None  # 实际中需要加载
        
    def query_kg(self, entity, relation=None):
        """查询知识图谱"""
        results = []
        
        if relation:
            # 查询特定关系
            for s, r, o in self.kg_triples:
                if s.lower() == entity.lower() and r.lower() == relation.lower():
                    results.append(o)
        else:
            # 获取实体所有信息
            for s, r, o in self.kg_triples:
                if s.lower() == entity.lower():
                    results.append((r, o))
        
        return results
    
    def answer_question(self, question):
        """回答基于知识图谱的问题"""
        # 简化的问题分析(实际中需要更复杂的NLP)
        entity = None
        relation = None
        
        if "爱因斯坦" in question:
            entity = "爱因斯坦"
            
            if "出生" in question:
                relation = "出生于" if "哪里" in question else "出生日期"
            elif "职业" in question:
                relation = "职业"
            elif "获得" in question and "奖" in question:
                relation = "获得"
        
        if not entity:
            return "无法识别问题中的实体"
        
        if relation:
            # 查询特定关系
            answers = self.query_kg(entity, relation)
            if answers:
                return f"{entity}的{relation}是{', '.join(answers)}"
            else:
                return f"未找到{entity}的{relation}信息"
        else:
            # 返回实体所有信息
            all_info = self.query_kg(entity)
            if all_info:
                results = [f"{r}: {o}" for r, o in all_info]
                return f"{entity}的信息：\n" + "\n".join(results)
            else:
                return f"未找到关于{entity}的信息"

# 使用示例
def test_kg_qa():
    kg_qa = KnowledgeGraphQA()
    
    questions = [
        "爱因斯坦出生在哪里？",
        "爱因斯坦的职业是什么？",
        "爱因斯坦什么时候出生的？",
        "爱因斯坦获得了什么奖项？"
    ]
    
    for question in questions:
        answer = kg_qa.answer_question(question)
        print(f"问题: {question}")
        print(f"答案: {answer}")
        print("-" * 50)