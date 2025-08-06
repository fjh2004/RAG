目前已实现的核心功能
  多模态文档管理
    支持文本（TXT、PDF、DOCX 等）和图像（PNG、JPG 等）的上传、解析、分块和向量入库（api.py的/documents上传接口、chroma_manager.py的add_documents）。
    提供文档列表查询、更新、清空等管理接口（api.py的GET /documents、PUT /documents/{file_id}、DELETE /documents）。
  多模态检索与问答
    支持纯文本、纯图像、图文混合查询，通过向量数据库（Chroma）实现相似性检索，并结合 BERT 模型重排序优化结果（chroma_manager.py的query_text/query_image/query_multimodal）。
    集成知识图谱查询（基于 Neo4j），可返回实体关系结果（neo4j_connector.py的query_kg、api.py的/query接口）。
辅助功能
    缓存管理：对高频查询结果进行缓存，带过期时间（cache_manager.py）。
    大模型集成：支持调用 OpenAI API 或本地模型生成回答（llm_manager.py）。
    系统监控：提供状态查询（文档数量、最后更新时间）和健康检查接口（api.py的/status、main.py的/health）。
    错误处理：全局异常捕获和日志记录（main.py的global_exception_handler）。
