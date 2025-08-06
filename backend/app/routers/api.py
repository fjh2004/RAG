from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
from pydantic import BaseModel
from datetime import datetime
import os
import logging
from datetime import datetime
from utils.document_processor import DocumentProcessor
from utils.chroma_manager import ChromaManager
import zipfile
import tempfile

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据模型定义
class DocumentItem(BaseModel):
    id: str
    filename: str
    type: str
    upload_time: datetime

class QueryRequest(BaseModel):
    query_text: Optional[str] = None  # 文本查询（可选）
    image_path: Optional[str] = None  # 图片路径（可选，支持本地路径）
    kg_query: Optional[str] = None  # 知识图谱查询语句（可选）
    top_k: int = 5
    score_threshold: float = 0.7  # 相关性阈值（转换为距离阈值使用）

class QueryResponse(BaseModel):
    results: List[Dict]
    entities: List[Dict] = []  # 知识图谱实体列表

class SystemStatus(BaseModel):
    vector_db: str
    document_count: int
    last_updated: datetime

# 初始化路由和工具类
router = APIRouter(prefix="/api/v1", tags=["Document Management", "QA", "System Monitoring"])
processor = DocumentProcessor()
chroma_mgr = ChromaManager()

@router.post("/documents", status_code=201, summary="上传多格式文档")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    支持格式: TXT, PDF, DOCX, DOC, Markdown, PNG, JPG, JPEG
    - 文本文件直接解析
    - 图片文件通过OCR提取文字和特征
    - 图片型PDF自动转为图片进行OCR处理
    """
    all_results = []
    for file in files:
        try:
            # 生成唯一保存路径
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            file_path = f"uploads/{timestamp}_{file.filename}"
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # 保存上传文件
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # 根据文件类型处理文档
            file_ext = os.path.splitext(file.filename)[1].lower()
            document = None
                   
            # 处理图片文件（PNG/JPG/JPEG）
            if file_ext in ['.png', '.jpg', '.jpeg']:
                image_result = processor.process_image(file_path)
                document = {
                    "text": image_result["text"],
                    "metadata": image_result["metadata"]
                }
            
            # 处理其他支持的文档格式（PDF/DOCX/DOC/Markdown/txt）
            else:
                document = processor.load_document(file_path)
            
            # 文档分块并入库
            chunks = processor.chunk_document(document)
            chroma_mgr.add_documents(chunks)
            
            all_results.append({
                "status": "success",
                "filename": file.filename,
                "file_id": file_path,
                "chunks": len(chunks),
                "type": document["metadata"]["type"]
            })
        
        except Exception as e:
            logger.error(f"处理文件 {file.filename} 失败: {str(e)}")
            all_results.append({
                "status": "failed",
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content=all_results)

@router.post("/query", response_model=QueryResponse, summary="多模态智能问答接口")
async def query_documents(request: QueryRequest):
    """
    RAG核心问答接口，支持：
    - 纯文本查询
    - 纯图片查询
    - 图文混合查询
    - 相关性过滤与重排序
    """
    if not request.query_text and not request.image_path:
        raise HTTPException(
            status_code=400,
            detail="必须提供query_text或image_path中的至少一项"
        )
    
    try:
        # 调用多模态查询方法
        results = chroma_mgr.query_multimodal(
            query_text=request.query_text,
            image_path=request.image_path,
            top_k=request.top_k,
            rerank=True
        )

        # 知识图谱查询
        kg_entities = []
        if request.kg_query:
            kg_entities = chroma_mgr.query_kg(
                query_text=request.query_text,
                kg_query=request.kg_query,
                top_k=request.top_k
            )
        
        # 基于阈值过滤结果（L2距离越小越相关）
        distance_threshold = 1 - request.score_threshold  # 转换相关性阈值为距离阈值
        filtered_results = [r for r in results if r["score"] <= distance_threshold]
        
        # 格式化返回结果
        return {
            "results": [
                {
                    "id": str(hash(r["text"])),
                    "text": r["text"],
                    "source": r["metadata"]["source"],
                    "original_score": r.get("original_score", r["score"]),  # 原始距离得分
                    "rerank_score": r.get("rerank_score")  # 重排序后的相关性得分（越高越好）
                }
                for r in filtered_results[:request.top_k]  # 确保不超过top_k
            ],
            "entities": kg_entities  # 知识图谱实体结果
        }
    except Exception as e:
        logger.error(f"查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status", response_model=SystemStatus, summary="系统状态查询")
async def get_system_status():
    """查询系统当前状态，包括向量库信息、文档数量和最后更新更新时间"""
    try:
        # 获取文档总数
        document_count = chroma_mgr.collection.count()
        
        # 简化处理：实际应用中应从数据库记录最后更新时间
        last_updated = datetime.now()
        
        return {
            "vector_db": "Chroma (multimodal_docs)",
            "document_count": document_count,
            "last_updated": last_updated
        }
    except Exception as e:
        logger.error(f"获取系统状态失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
#批量导入接口
@router.post("/documents/batch", status_code=201, summary="批量导入文档（支持ZIP压缩包）")
async def batch_upload_documents(
    zip_file: UploadFile = File(...),
    recursive: bool = False  # 是否递归处理子目录
):
    """
    批量导入文档，支持上传ZIP压缩包，自动解压并处理内部所有支持格式的文件
    支持格式: TXT, PDF, DOCX, DOC, Markdown, PNG, JPG, JPEG
    """
    if not zip_file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="仅支持ZIP格式压缩包")
    
    all_results = []
    
    try:
        # 创建临时目录存放解压文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存上传的ZIP文件
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(await zip_file.read())
            
            # 解压ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # 遍历处理所有文件
            for root, dirs, files in os.walk(temp_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_ext = os.path.splitext(file_name)[1].lower()
                    
                    # 跳过临时ZIP文件和不支持的格式
                    if file_path == zip_path:
                        continue
                    if file_ext not in ['.txt', '.pdf', '.docx', '.doc', '.md', '.png', '.jpg', '.jpeg']:
                        all_results.append({
                            "status": "skipped",
                            "filename": file_name,
                            "reason": f"不支持的文件格式: {file_ext}"
                        })
                        continue
                    
                    try:
                        # 处理文档
                        document = None
                        if file_ext in ['.png', '.jpg', '.jpeg']:
                            image_result = processor.process_image(file_path)
                            document = {
                                "text": image_result["text"],
                                "metadata": image_result["metadata"]
                            }
                        else:
                            document = processor.load_document(file_path)
                        
                        # 分块入库
                        chunks = processor.chunk_document(document)
                        chroma_mgr.add_documents(chunks)
                        
                        all_results.append({
                            "status": "success",
                            "filename": file_name,
                            "file_path": os.path.relpath(file_path, temp_dir),
                            "chunks": len(chunks),
                            "type": document["metadata"]["type"]
                        })
                    except Exception as e:
                        logger.error(f"处理文件 {file_name} 失败: {str(e)}")
                        all_results.append({
                            "status": "failed",
                            "filename": file_name,
                            "error": str(e)
                        })
                
                # 如果不递归处理，只处理顶层目录
                if not recursive:
                    break
    
    except Exception as e:
        logger.error(f"批量导入失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return JSONResponse(content=all_results)

# 在api.py中添加知识库管理接口
@router.get("/documents", summary="获取文档列表")
async def list_documents(
    document_type: Optional[str] = None,
    page: int = 1,
    page_size: int = 20
):
    """
    获取知识库中的文档列表，支持按类型筛选和分页
    """
    try:
        # 构建筛选条件
        filter_conditions = {}
        if document_type:
            filter_conditions["document_type"] = document_type
        
        # 获取文档元数据（实际项目中应优化查询性能）
        documents = chroma_mgr.get_documents_metadata(
            filter_conditions=filter_conditions,
            page=page,
            page_size=page_size
        )
        
        return JSONResponse(content={
            "total": documents["total"],
            "page": page,
            "page_size": page_size,
            "documents": documents["items"]
        })
    except Exception as e:
        logger.error(f"获取文档列表失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{file_id}", summary="删除指定文档")
async def delete_document(file_id: str):
    """
    根据文件ID删除知识库中的文档及其所有分块
    """
    try:
        # 删除所有关联的分块
        deleted_count = chroma_mgr.delete_by_metadata({"source": file_id})
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="文档不存在")
        
        return JSONResponse(content={
            "status": "success",
            "message": f"成功删除文档 {file_id} 及其 {deleted_count} 个分块"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
#知识库管理功能实现
@router.put("/documents/{file_id}", summary="更新文档")
async def update_document(
    file_id: str,
    file: UploadFile = File(...)
):
    """
    更新知识库中的文档（先删除旧版本，再添加新版本）
    """
    try:
        # 先删除旧文档
        chroma_mgr.delete_by_metadata({"source": file_id})
        
        # 处理新文档
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        new_file_path = f"uploads/{timestamp}_{file.filename}"
        os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
        
        with open(new_file_path, "wb") as f:
            f.write(await file.read())
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        document = None
        
        if file_ext in ['.png', '.jpg', '.jpeg']:
            image_result = processor.process_image(new_file_path)
            document = {
                "text": image_result["text"],
                "metadata": image_result["metadata"]
            }
        else:
            document = processor.load_document(new_file_path)
        
        chunks = processor.chunk_document(document)
        chroma_mgr.add_documents(chunks)
        
        return JSONResponse(content={
            "status": "success",
            "old_file_id": file_id,
            "new_file_id": new_file_path,
            "chunks": len(chunks),
            "type": document["metadata"]["type"]
        })
    except Exception as e:
        logger.error(f"更新文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents", summary="清空文档数据库")
async def clear_documents():
    """清空向量数据库中的所有文档（用于测试或重置）"""
    try:
        chroma_mgr.clear_database()
        return JSONResponse(content={"status": "success", "message": "所有文档已清空"})
    except Exception as e:
        logger.error(f"清空文档失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))