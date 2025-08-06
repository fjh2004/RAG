from py2neo import Graph, Node, Relationship
import os
from dotenv import load_dotenv
from typing import List, Dict
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class Neo4jConnector:
    def __init__(self):
        self.graph = Graph(
            os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            auth=(
                os.getenv('NEO4J_USER', 'neo4j'),
                os.getenv('NEO4J_PASSWORD', 'password')
            )
        )
        self._create_constraints()

    def _create_constraints(self):
        # 创建唯一性约束
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (r:Relation) REQUIRE r.id IS UNIQUE")

    def merge_entity(self, entity: Dict) -> Node:
        """合并实体节点（存在则更新属性）"""
        node = Node(
            "Entity",
            id=entity['id'],
            name=entity['name'],
            type=entity['type'],
            source=entity.get('source', ''),
            properties=entity.get('properties', {})
        )
        self.graph.merge(node, "Entity", "id")
        return node

    def create_relationship(self, head: Node, tail: Node, rel_type: str, properties: Dict = {}) -> Relationship:
        """创建实体关系"""
        rel = Relationship(head, rel_type, tail, **properties)
        self.graph.merge(rel)
        return rel

    def query_kg(self, cypher: str, params: Dict = None) -> List[Dict]:
        """执行Cypher查询"""
        try:
            result = self.graph.run(cypher, params)
            return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Cypher查询失败: {str(e)}")
            raise

    def batch_merge_entities(self, entities: List[Dict]):
        """批量合并实体（事务处理）"""
        tx = self.graph.begin()
        try:
            for entity in entities:
                node = Node("Entity", **entity)
                tx.merge(node, "Entity", "id")
            tx.commit()
        except Exception as e:
            tx.rollback()
            raise

    def clear_graph(self):
        """清空图谱数据（仅开发环境使用）"""
        self.graph.run("MATCH (n) DETACH DELETE n")