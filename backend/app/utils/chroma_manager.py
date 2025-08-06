import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch
import clip
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizer, BertForSequenceClassification
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaManager:
    def __init__(self, pca_components: int = 1024):
        # 初始化Chroma持久化客户端
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # 加载文本嵌入模型(BAAI/bge-large-en-v1.5)，输出1024维向量
        self.text_embedder = SentenceTransformer('BAAI/bge-large-en-v1.5')
        
        # 配置计算设备(GPU优先)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 加载CLIP模型(图像编码器)，输出768维向量
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # 创建或获取集合(所有文档最终均为1024维向量)
        self.collection = self.client.get_or_create_collection(
            name="multimodal_docs",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        
        # 初始化PCA模型：将图片文档的1792维向量(1024文本+768图像)降维至1024维
        self.pca_components = pca_components  # 目标维度1024
        self.pca = PCA(n_components=pca_components)
        self.scaler = StandardScaler()  # 标准化PCA输入
        self.pca_trained = False  # 标记PCA是否训练完成
        
        # 初始化重排序模型(用于优化检索结果)
        self.rerank_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.rerank_model = BertForSequenceClassification.from_pretrained(
            "bert-base-chinese", 
            num_labels=2
        ).to(self.device)
        
        # 尝试加载预训练的PCA模型(针对图片文档的1792→1024降维)
        self._load_pca_model()

    def _train_pca(self, multimodal_embeddings: np.ndarray) -> None:
        """训练PCA模型：用图片文档的1792维拼接向量作为训练数据"""
        if self.pca_trained:
            return
            
        # 标准化数据(消除不同维度量级影响)
        scaled_embeddings = self.scaler.fit_transform(multimodal_embeddings)
        
        # 训练PCA：保留1024个主成分，最大化保留原始信息
        self.pca.fit(scaled_embeddings)
        self.pca_trained = True
        
        # 保存PCA模型参数(方便下次加载)
        np.save("pca_1792to1024_components.npy", self.pca.components_)
        np.save("pca_1792to1024_mean.npy", self.pca.mean_)
        np.save("scaler_1792_mean.npy", self.scaler.mean_)
        np.save("scaler_1792_scale.npy", self.scaler.scale_)
        
        # 计算保留的信息量(方差解释率)
        explained_variance = np.sum(self.pca.explained_variance_ratio_)
        logger.info(f"PCA训练完成(1792→1024)，保留方差: {explained_variance:.4f} (越高越好，建议>0.9)")

    def _load_pca_model(self) -> None:
        """加载预训练的PCA模型(针对1792→1024降维)"""
        try:
            if (os.path.exists("pca_1792to1024_components.npy") and 
                os.path.exists("pca_1792to1024_mean.npy") and
                os.path.exists("scaler_1792_mean.npy") and
                os.path.exists("scaler_1792_scale.npy")):
                
                self.pca.components_ = np.load("pca_1792to1024_components.npy")
                self.pca.mean_ = np.load("pca_1792to1024_mean.npy")
                self.scaler.mean_ = np.load("scaler_1792_mean.npy")
                self.scaler.scale_ = np.load("scaler_1792_scale.npy")
                self.pca_trained = True
                logger.info("PCA模型(1792→1024)加载成功")
        except Exception as e:
            logger.warning(f"加载PCA模型失败: {str(e)}, 将在首次处理图片文档时训练")

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        """生成文本嵌入(1024维，不降维，保留完整语义)"""
        # 直接用BGE模型生成1024维向量(文本占比高，优先保证文本精度)
        return self.text_embedder.encode(texts).tolist()

    def embed_image(self, image_path: str) -> List[float]:
        """生成图像嵌入(768维，CLIP模型输出)"""
        try:
            image = Image.open(image_path).convert("RGB")
            preprocessed = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():  # 关闭梯度计算，加速并节省内存
                features = self.clip_model.encode_image(preprocessed)
                
            return features.cpu().numpy().flatten().tolist()  # 768维向量
        except Exception as e:
            logger.error(f"图像嵌入生成失败: {str(e)}")
            raise

    def add_documents(self, documents: List[Dict]) -> None:
        """
        添加文档到向量数据库：
        - 纯文本文档：直接使用1024维文本向量
        - 图片文档：文本向量(1024) + 图像向量(768) → 1792维 → PCA降维至1024维
        """
        try:
            # 生成唯一ID(基于文本哈希，避免重复添加)
            ids = [str(hash(doc['text'])) for doc in documents]
            texts = [doc['text'] for doc in documents]
            metadatas = [doc['metadata'] for doc in documents]
            
            # 生成文本嵌入(1024维，所有文档均需文本嵌入)
            text_embeddings = self.embed_text(texts)  # 列表：每个元素是1024维向量
            
            # 处理最终嵌入(区分文本/图片文档)
            final_embeddings = []
            multimodal_embeddings_for_pca = []  # 收集图片文档的1792维向量，用于训练PCA
            
            for i, doc in enumerate(documents):
                if doc['metadata'].get('type') == 'image':
                    # 图片文档：拼接→降维
                    # 1. 生成图像嵌入(768维)
                    image_emb = self.embed_image(doc['metadata']['source'])
                    # 2. 拼接文本(1024)和图像(768)→1792维
                    combined_emb = np.concatenate([text_embeddings[i], image_emb])
                    multimodal_embeddings_for_pca.append(combined_emb)
                    
                    # 3. 若PCA未训练，先训练(用当前图片文档的1792维向量)
                    if not self.pca_trained:
                        logger.info("PCA未训练，使用图片文档的拼接向量训练...")
                        self._train_pca(np.array(multimodal_embeddings_for_pca))
                    
                    # 4. 应用PCA降维至1024维
                    scaled_emb = self.scaler.transform(combined_emb.reshape(1, -1))  # 标准化
                    reduced_emb = self.pca.transform(scaled_emb).flatten()  # 降维
                    final_embeddings.append(reduced_emb.tolist())
                else:
                    # 纯文本文档：直接使用1024维向量
                    final_embeddings.append(text_embeddings[i])
            
            # 添加到向量数据库(所有文档均为1024维)
            self.collection.add(
                ids=ids,
                embeddings=final_embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            logger.info(f"成功添加 {len(documents)} 个文档(文本{len(documents)-len(multimodal_embeddings_for_pca)}个，图片{len(multimodal_embeddings_for_pca)}个)")
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise

    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """用BERT重排序，优化检索结果精度"""
        if not results:
            return []
            
        scores = []
        for res in results:
            try:
                # 拼接查询和结果文本，计算相关性
                inputs = self.rerank_tokenizer(
                    query, 
                    res["text"], 
                    return_tensors="pt", 
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    logits = self.rerank_model(** inputs).logits
                    score = torch.softmax(logits, dim=1)[0][1].item()  # 相关概率
                scores.append(score)
            except Exception as e:
                logger.warning(f"重排序失败: {str(e)}, 使用原始得分")
                scores.append(1.0 - res["score"])  # 用距离倒数替代
        
        # 按得分排序
        sorted_pairs = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [{"text": p[0]["text"], "metadata": p[0]["metadata"], 
                 "original_score": p[0]["score"], "rerank_score": p[1]} 
                for p in sorted_pairs]

    def query_kg(self, query_text: str, kg_query: str, top_k: int) -> List[Dict]:
        """执行知识图谱查询"""
        try:
            return self.kg_processor.query_kg(
                entity=query_text,
                query_pattern=kg_query,
                top_k=top_k
            )
        except Exception as e:
            logger.error(f"知识图谱查询失败: {str(e)}")
            return []

    def query_text(self, 
                  query_text: str, 
                  top_k: int = 5, 
                  filter_conditions: Optional[Dict] = None,
                  rerank: bool = True) -> List[Dict]:
        """文本查询：生成1024维向量，直接检索"""
        try:
            # 生成查询向量(1024维，与文档向量维度一致)
            query_embedding = self.embed_text([query_text])[0]
            
            # 检索(支持元数据过滤，如只查PDF)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3,  # 多取3倍结果用于重排序
                where=filter_conditions
            )
            
            # 整理结果
            formatted_results = [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": score  # L2距离，越小越相关
                }
                for doc, meta, score in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
            # 重排序优化
            if rerank:
                formatted_results = self._rerank_results(query_text, formatted_results)
            
            return formatted_results[:top_k]
        except Exception as e:
            logger.error(f"文本查询失败: {str(e)}")
            raise

    #图查询应用一般是电商 查相似产品、相似图片等等
    def query_image(self, 
                   image_path: str, 
                   top_k: int = 5, 
                   filter_conditions: Optional[Dict] = None,
                   rerank: bool = True) -> List[Dict]:
        """图像查询：生成1792维拼接向量→降维至1024维，再检索"""
        try:
            # 1. 生成图像嵌入(768维)
            image_emb = self.embed_image(image_path)
            # 2. 用空文本生成1024维向量(代表"无文本信息")
            empty_text_emb = self.embed_text([""])[0]  # 空文本的1024维向量
            # 3. 拼接→1792维
            combined_emb = np.concatenate([empty_text_emb, image_emb])
            # 4. PCA降维至1024维(与文档向量一致)
            scaled_emb = self.scaler.transform(combined_emb.reshape(1, -1))
            query_embedding = self.pca.transform(scaled_emb).flatten().tolist()
            
            # 检索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3,
                where=filter_conditions
            )
            
            # 整理结果
            formatted_results = [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": score
                }
                for doc, meta, score in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
            # 重排序(用图像OCR文本作为查询依据)
            if rerank:
                from utils.document_processor import DocumentProcessor  # 导入OCR工具
                ocr_text = DocumentProcessor().process_image(image_path)["text"]
                query_for_rerank = f"与图像内容'{ocr_text}'相关的内容"
                formatted_results = self._rerank_results(query_for_rerank, formatted_results)
            
            return formatted_results[:top_k]
        except Exception as e:
            logger.error(f"图像查询失败: {str(e)}")
            raise
    
    def query_multimodal(self, 
                    query_text: Optional[str] = None, 
                    image_path: Optional[str] = None, 
                    top_k: int = 5, 
                    filter_conditions: Optional[Dict] = None,
                    rerank: bool = True) -> List[Dict]:
        # """
        # 多模态混合查询（文字+图像），采用向量平等融合（各占50%权重）
        # - 支持纯文本、纯图像、文字+图像三种输入模式
        # - 向量融合公式：V = (V_text * 0.5) + (V_image * 0.5)
        # """
        # 校验输入合法性（至少提供一种输入）
        
        if not query_text and not image_path:
            raise ValueError("至少提供文字查询或图像路径")
        
        if query_text and not image_path:
            return self.query_text(
                query_text=query_text,
                top_k=top_k,
                filter_conditions=filter_conditions,
                rerank=rerank
                )
        elif image_path and not query_text:
            return self.query_image(
                image_path=image_path,
                top_k=top_k,
                filter_conditions=filter_conditions,
                rerank=rerank
                )

        try:
            # 1. 生成文本向量（若提供文字）
            if query_text:
                # 文本向量：1024维（纯文本嵌入，不降维）
                text_emb = np.array(self.embed_text([query_text])[0], dtype=np.float32)
            else:
                # 无文字时，文本向量用全0向量（1024维）
                text_emb = np.zeros(1024, dtype=np.float32)
            
            # 2. 生成图像向量（若提供图像）
            if image_path:
                # 图像向量处理：768维图像嵌入 → 拼接空文本向量 → PCA降维至1024维
                image_emb_768 = np.array(self.embed_image(image_path), dtype=np.float32)
                empty_text_emb = np.array(self.embed_text([""])[0], dtype=np.float32)  # 空文本的1024维向量
                combined_1792 = np.concatenate([empty_text_emb, image_emb_768])  # 1792维拼接向量
                # PCA降维至1024维（与文本向量维度一致）
                scaled_emb = self.scaler.transform(combined_1792.reshape(1, -1))
                image_emb = self.pca.transform(scaled_emb).flatten().astype(np.float32)
            else:
                # 无图像时，图像向量用全0向量（1024维）
                image_emb = np.zeros(1024, dtype=np.float32)
            
            # 3. 向量平等融合（各占50%权重）
            # 确保两者维度一致（均为1024维）
            if len(text_emb) != len(image_emb):
                raise ValueError(f"向量维度不匹配：文本{len(text_emb)}维，图像{len(image_emb)}维")
            
            fused_embedding = (text_emb * 0.5 + image_emb * 0.5).tolist()  # 融合后的1024维向量
            
            # 4. 用融合向量执行检索
            results = self.collection.query(
                query_embeddings=[fused_embedding],
                n_results=top_k * 3,  # 多取3倍结果用于重排序
                where=filter_conditions
            )
            
            # 5. 整理初步结果
            formatted_results = [
                {
                    "text": doc,
                    "metadata": meta,
                    "score": score  # L2距离，越小越相关
                }
                for doc, meta, score in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
            
            # 6. 重排序优化（结合文字和图像语义）
            if rerank:
                # 构建重排序用的查询文本（融合文字和图像OCR内容）
                if image_path:
                    # 提取图像OCR文本作为补充
                    from utils.document_processor import DocumentProcessor
                    ocr_processor = DocumentProcessor()
                    image_ocr_text = ocr_processor.process_image(image_path)["text"]
                    rerank_query = f"{query_text or ''}。图像内容：{image_ocr_text}"
                else:
                    rerank_query = query_text  # 纯文本时直接用输入文字
                
                formatted_results = self._rerank_results(rerank_query, formatted_results)
            
            return formatted_results[:top_k]
    
        except Exception as e:
            logger.error(f"多模态查询失败: {str(e)}")
            raise
    
    # 支持知识库管理
    def get_documents_metadata(
        self, 
        filter_conditions: Optional[Dict] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict:
        """获取文档元数据列表，支持筛选和分页"""
        try:
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 查询所有符合条件的文档
            results = self.collection.get(
                where=filter_conditions,
                include=["metadatas", "documents"]
            )
            
            # 提取唯一文档（去重）
            unique_documents = {}
            for meta, doc in zip(results["metadatas"], results["documents"]):
                source = meta["source"]
                if source not in unique_documents:
                    unique_documents[source] = {
                        "source": source,
                        "type": meta["document_type"],
                        "first_chunk": doc,
                        "chunk_count": 1
                    }
                else:
                    unique_documents[source]["chunk_count"] += 1
            
            # 转换为列表并分页
            doc_list = list(unique_documents.values())
            total = len(doc_list)
            paginated = doc_list[offset:offset + page_size]
            
            return {
                "total": total,
                "items": paginated
            }
        except Exception as e:
            logger.error(f"获取文档元数据失败: {str(e)}")
            raise

    def delete_by_metadata(self, filter_conditions: Dict) -> int:
        """根据元数据条件删除文档"""
        try:
            # 获取符合条件的ID
            results = self.collection.get(where=filter_conditions, include=[])
            if not results["ids"]:
                return 0
            
            # 删除文档
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        except Exception as e:
            logger.error(f"按条件删除文档失败: {str(e)}")
            raise

    def clear_database(self) -> None:
        """清空数据库(用于测试或重置)"""
        try:
            self.client.delete_collection(name="multimodal_docs")
            self.collection = self.client.get_or_create_collection(
                name="multimodal_docs",
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            logger.info("数据库已清空")
        except Exception as e:
            logger.error(f"清空数据库失败: {str(e)}")
            raise