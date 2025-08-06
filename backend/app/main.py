from routers.api import router  
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# 优先加载环境变量
load_dotenv()

# 配置日志（确保全局唯一初始化）
def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ]
    )

configure_logging()
logger = logging.getLogger(__name__)

# 初始化上传目录
def init_upload_dir():
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
    try:
        UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
        logger.info(f"上传目录初始化成功: {UPLOAD_DIR}")
        return UPLOAD_DIR
    except PermissionError:
        logger.error(f"无法创建上传目录 {UPLOAD_DIR}，权限不足")
        raise
    except Exception as e:
        logger.error(f"上传目录初始化失败: {str(e)}")
        raise

UPLOAD_DIR = init_upload_dir()

# 初始化FastAPI应用
app = FastAPI(
    title="MultiModalDocQA API",
    description="""
    多模态文档问答系统后端接口，支持：
    - 批量导入文档（ZIP压缩包）
    - 知识库管理（查询、删除、更新文档）
    - 多模态问答（文本/图片输入）
    """,
    version="0.1.0",
    contact={"name": "Dev Team", "email": "dev@example.com"}
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOW_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"请求异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误，请稍后再试"}
    )

# 包含路由
app.include_router(router)

# 健康检查接口
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "multimodal_doc_qa",
        "version": "0.1.0",
        "upload_dir": str(UPLOAD_DIR)
    }

# 启动配置
if __name__ == "__main__":
    import uvicorn
    is_dev = os.getenv("ENVIRONMENT", "dev") == "dev"
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=is_dev,
        log_level="debug" if is_dev else "info"
    )