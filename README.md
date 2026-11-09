# 多模态文档问答后端

一个把文档和图片做成可检索资料库的 Python 后端原型。它将上传的资料解析、分块并建立向量索引，支持按文字、图片或图文组合检索相关片段，也可以补充知识图谱中的实体关系。

## 主要功能

- **资料管理**支持解析 TXT、Markdown、PDF、Word 和常见图片，提取文字并分块。接口包含上传、ZIP 批量导入、查询列表、更新和删除。
- **多模态检索**使用 Chroma 存储向量，结合文本嵌入、CLIP 图像特征和 BERT 重排序，返回匹配片段、来源和评分。
- **知识图谱**使用 Neo4j 存储和查询实体关系。查询接口可以在检索片段之外返回关联实体。
- **问答生成**保留了大模型调用、提示词和会话管理模块。当前的 `/api/v1/query` 接口返回检索片段和实体，生成答案的逻辑在独立模块中。

## 代码位置

| 文件 | 职责 |
| --- | --- |
| [`backend/app/main.py`](backend/app/main.py) | FastAPI 入口、健康检查和异常处理 |
| [`backend/app/routers/api.py`](backend/app/routers/api.py) | 资料管理、检索和系统状态接口 |
| [`document_processor.py`](backend/app/utils/document_processor.py) | 文件解析、OCR 和文本分块 |
| [`chroma_manager.py`](backend/app/utils/chroma_manager.py) | 向量索引、多模态检索和重排序 |
| [`kg_processor.py`](backend/app/utils/kg_processor.py) | 实体与关系抽取、图谱查询 |
| [`llm_manager.py`](backend/app/utils/llm_manager.py) | 调用 OpenAI API 或本地模型生成文本 |
