## 核心功能实现

### 1. 多模态文档管理
- 支持文本（TXT、PDF、DOCX 等）和图像（PNG、JPG 等）的：
  - 上传与解析
  - 内容分块处理
  - 向量入库存储  
  （对应实现：`api.py` 的 `/documents` 上传接口、`chroma_manager.py` 的 `add_documents` 方法）

- 提供完整的文档管理接口：
  - 文档列表查询（`GET /documents`）
  - 单文档更新（`PUT /documents/{file_id}`）
  - 全量文档清空（`DELETE /documents`）  


### 2. 多模态检索与问答
- 多类型查询支持：
  - 纯文本检索
  - 纯图像检索
  - 图文混合检索  
  （技术实现：基于向量数据库 Chroma 实现相似性检索，结合 BERT 模型重排序优化结果，对应 `chroma_manager.py` 的 `query_text`/`query_image`/`query_multimodal` 方法）

- 知识图谱集成：
  - 基于 Neo4j 实现实体关系查询
  - 支持返回结构化的实体关联结果  
  （对应实现：`neo4j_connector.py` 的 `query_kg` 方法、`api.py` 的 `/query` 接口）


## 辅助功能

| 功能类别       | 具体说明                                                                 | 实现位置                          |
|----------------|--------------------------------------------------------------------------|-----------------------------------|
| 缓存管理       | 对高频查询结果进行缓存，支持设置过期时间，提升重复查询效率               | `cache_manager.py`                |
| 大模型集成     | 支持调用 OpenAI API 或本地部署模型生成问答结果，灵活适配不同算力环境     | `llm_manager.py`                  |
| 系统监控       | 提供状态查询（文档总量、最后更新时间）和健康检查接口，便于运维监控       | `api.py` 的 `/status` 接口、`main.py` 的 `/health` 接口 |
| 错误处理       | 全局异常捕获机制，配合详细日志记录，便于问题排查与调试                   | `main.py` 的 `global_exception_handler` 方法 |
