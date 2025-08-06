import spacy
from neo4j import GraphDatabase
import re
from typing import List, Tuple, Dict

# 加载NLP模型（需先下载：python -m spacy download zh_core_web_md）
nlp = spacy.load("zh_core_web_md")

class KGProcessor:
    def __init__(self, neo4j_uri="bolt://localhost:7687", username="neo4j", password="password"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def extract_entities(self, text: str) -> List[Dict]:
        """结合spaCy和规则模型抽取实体"""
        doc = nlp(text)
        entities = []

        # spaCy实体识别
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        # 添加领域特定规则
        tech_terms = re.findall(r'\b[A-Z0-9]{2,}(?:\s+[A-Z0-9]{2,})+\b', text)
        entities.extend([{'text': term, 'type': 'TECH_TERM', 'start': text.find(term), 'end': text.find(term)+len(term)} 
                        for term in tech_terms])

        return self._merge_entities(entities)

    def extract_relations(self, text: str, entities: List[Dict]) -> List[Dict]:
        """基于句法分析和规则的关系抽取"""
        relations = []
        doc = nlp(text)

        # 依存句法分析
        for token in doc:
            if token.dep_ in ('dobj', 'nsubj'):
                subj = next((e for e in entities if e['start'] <= token.head.idx <= e['end']), None)
                obj = next((e for e in entities if e['start'] <= token.idx <= e['end']), None)
                if subj and obj:
                    relations.append({
                        'source': subj['text'],
                        'target': obj['text'],
                        'type': token.dep_,
                        'context': token.sent.text
                    })

        # 自定义规则模式
        rule_patterns = [
            (r'(\w+)[是｜为]([\u4e00-\u9fa5]+)的([\u4e00-\u9fa5]+)', '属性关系'),
            (r'(\w+)导致(\w+)', '因果关系'),
            (r'(\w+)采用(\w+)技术', '技术应用')
        ]

        for pattern, rel_type in rule_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                relations.append({
                    'source': match.group(1),
                    'target': match.group(2),
                    'type': rel_type,
                    'context': match.group(0)
                })

        return relations

    def add_triples_to_kg(self, triples: List[Tuple[str, str, str]]):
        """将三元组存入Neo4j"""
        with self.driver.session() as session:
            for subj, rel, obj in triples:
                session.run(
                    "MERGE (s:Entity {name: $subj}) "
                    "MERGE (o:Entity {name: $obj}) "
                    "MERGE (s)-[r:RELATION {type: $rel}]->(o)",
                    subj=subj, obj=obj, rel=rel
                )
            session.commit()
    
    def query_kg(self, entity: str, top_k: int = 3) -> List[Dict]:
        """查询实体相关的三元组"""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:Entity {name: $entity})-[r:RELATION]->(o:Entity) "
                "RETURN s.name AS subject, r.type AS relation, o.name AS object "
                "LIMIT $top_k",
                entity=entity, top_k=top_k
            )
            return [{"subject": r["subject"], "relation": r["relation"], "object": r["object"]} 
                    for r in result.data()]