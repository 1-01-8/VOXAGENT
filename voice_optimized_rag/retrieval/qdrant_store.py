"""
Qdrant 向量数据库适配器 —— 生产环境云端检索

与 FAISSVectorStore 提供完全相同的接口，可以无缝替换。
区别在于：FAISS 在本地内存中检索（无网络延迟），
Qdrant 需要经过真实的网络请求（云端 ~110ms），
更贴近生产环境的实际延迟，适合用于基准测试和生产部署。

支持两种部署方式：
- Qdrant Cloud（远程，需要 URL + API Key）
- 本地 Docker（localhost:6333，无需 API Key）
"""

from __future__ import annotations

from dataclasses import dataclass  # 数据类装饰器（虽然此文件未直接使用，但保持导入一致性）
from uuid import uuid4             # 生成全局唯一 ID，作为每个向量点的标识符

import numpy as np

# SearchResult 是与 FAISSVectorStore 共用的返回数据结构，保证接口兼容
from voice_optimized_rag.retrieval.vector_store import SearchResult
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("qdrant_store")


class QdrantVectorStore:
    """
    基于 Qdrant 的向量数据库 —— FAISSVectorStore 的生产级替代品

    接口与 FAISSVectorStore 完全一致（add_documents / search / size），
    Memory Router 可以通过配置无缝切换，无需修改其他代码。

    Qdrant 相比 FAISS 的优势：
    - 持久化存储（重启后数据不丢失）
    - 支持过滤条件（按 metadata 字段过滤）
    - 云端部署，多实例共享同一知识库
    - 支持实时增删改查

    Args:
        dimension: 向量维度（必须与 Embedding 模型输出维度一致）
        url: Qdrant 服务地址（云端如 "https://xxx.cloud.qdrant.io"，本地为 "http://localhost:6333"）
        api_key: Qdrant Cloud 的认证密钥（本地部署不需要）
        collection_name: Qdrant 中的集合名称（类似数据库中的"表"）
    """

    def __init__(
        self,
        dimension: int,
        url: str = "http://localhost:6333",   # 默认本地 Docker 地址
        api_key: str | None = None,
        collection_name: str = "voice_rag",
    ) -> None:
        # 延迟导入：只有使用 Qdrant 时才需要安装 qdrant-client
        # 避免未安装时整个项目报错（可选依赖）
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError:
            raise ImportError("Install qdrant-client: pip install qdrant-client")

        self._dimension = dimension
        self._collection = collection_name   # 保存集合名，后续所有操作都需要指定
        self._document_version = 0

        # ── 连接 Qdrant 服务 ──
        kwargs: dict = {"url": url, "timeout": 30}  # timeout=30s，云端网络可能较慢
        if api_key:
            kwargs["api_key"] = api_key   # 云端认证，本地无需
        self._client = QdrantClient(**kwargs)

        # ── 初始化集合（不存在则创建，存在则复用）──
        # get_collections() 返回所有集合列表，提取名称做存在性检查
        collections = [c.name for c in self._client.get_collections().collections]
        if collection_name not in collections:
            # 创建新集合，指定向量维度和距离度量方式
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,           # 向量维度，必须与 Embedding 模型一致
                    distance=Distance.COSINE, # 使用余弦距离（语义相似度的标准度量）
                ),
            )
            logger.info(f"Created Qdrant collection: {collection_name}")
        else:
            # 集合已存在（如重启后），直接复用，数据不会丢失
            logger.info(f"Using existing Qdrant collection: {collection_name}")

    @property
    def size(self) -> int:
        """返回集合中当前存储的向量点总数"""
        info = self._client.get_collection(self._collection)
        return info.points_count or 0  # points_count 可能为 None，兜底返回 0

    @property
    def document_version(self) -> int:
        return self._document_version

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        """
        将文档块及其向量批量写入 Qdrant

        流程：校验输入 → L2 归一化（余弦相似度要求）→ 构建 PointStruct → 分批上传

        Args:
            texts: 文档块原文列表
            embeddings: 预计算好的向量矩阵，shape=(n, dimension)
            metadata: 每个文档块的元数据（来源文件名、分块 ID 等），可为 None
        """
        from qdrant_client.models import PointStruct  # 延迟导入，减少启动时间

        # 输入校验：文本数量必须与向量数量一致
        if len(texts) != embeddings.shape[0]:
            raise ValueError("texts and embeddings must have the same length")

        # ── L2 归一化（使余弦相似度等价于内积，提高搜索精度）──
        # keepdims=True 保持 (n,1) 形状，便于广播除法
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1          # 防止零向量除以 0（全零向量归一化后仍为零向量）
        normalized = (embeddings / norms).astype(np.float32)  # Qdrant 要求 float32

        # ── 构建 PointStruct 列表（Qdrant 的数据单元）──
        points = []
        for i, (text, emb) in enumerate(zip(texts, normalized)):
            meta = metadata[i] if metadata else {}
            points.append(PointStruct(
                id=str(uuid4()),          # 用 UUID 生成唯一 ID，避免 ID 冲突
                vector=emb.tolist(),      # numpy → Python list（Qdrant API 要求）
                payload={"text": text, **meta},  # payload 存储原文和元数据，检索时一起返回
            ))

        # ── 分批上传（避免单次请求体过大超时）──
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._client.upsert(
                collection_name=self._collection,
                points=batch,
                # upsert = update + insert：ID 存在则更新，不存在则插入
                # 支持文档更新场景（重新导入同 ID 文档时不会重复）
            )

        self.bump_version()
        logger.info(f"Added {len(texts)} documents to Qdrant (total: {self.size})")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        """
        在 Qdrant 中执行向量相似度检索（真实网络请求，~110ms）

        与 FAISSVectorStore.search 接口完全一致，返回相同的 SearchResult 列表。
        include_embeddings=True 时会返回文档块自身的向量，
        供 SemanticCache 以文档向量为键缓存（Slow Thinker 的核心机制）。

        Args:
            query_embedding: 查询向量，shape=(dimension,) 或 (1, dimension)
            top_k: 返回最相似的文档块数量
            include_embeddings: 是否在结果中包含文档块的向量

        Returns:
            SearchResult 列表，按相似度降序排列
        """
        # ── 归一化查询向量 ──
        query = query_embedding.reshape(-1).astype(np.float32)  # 展平为 1D
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm   # L2 归一化，与存储时保持一致

        # ── 发起 Qdrant 查询（真实网络 I/O）──
        results = self._client.query_points(
            collection_name=self._collection,
            query=query.tolist(),          # numpy → Python list
            limit=top_k,                   # 返回 top_k 个最相似结果
            with_vectors=include_embeddings,  # True 时返回文档向量（供缓存使用）
        )

        # ── 将 Qdrant 结果转换为统一的 SearchResult 格式 ──
        search_results = []
        for point in results.points:
            # 如果请求了向量且 point 包含向量，转换为 numpy 数组
            emb = None
            if include_embeddings and point.vector:
                emb = np.array(point.vector, dtype=np.float32)

            # 从 payload 中分离原文和元数据
            text = point.payload.get("text", "") if point.payload else ""
            # 过滤掉 "text" 键，剩余的都是元数据（来源、分块 ID 等）
            meta = {k: v for k, v in (point.payload or {}).items() if k != "text"}

            search_results.append(SearchResult(
                text=text,
                metadata=meta,
                score=point.score,                       # Qdrant 返回的余弦相似度分数
                index=hash(point.id) % (2**31),          # 将 UUID 字符串哈希为整数索引
                                                         # （FAISSVectorStore 用整数索引，保持接口兼容）
                embedding=emb,
                dense_score=point.score,
                retrieval_source="dense",
            ))

        return search_results

    def list_documents(self) -> list[SearchResult]:
        """Scroll all documents for sparse indexing and fusion sync."""
        documents: list[SearchResult] = []
        offset = None
        index = 0

        while True:
            points, next_offset = self._client.scroll(
                collection_name=self._collection,
                offset=offset,
                limit=256,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                text = point.payload.get("text", "") if point.payload else ""
                meta = {k: v for k, v in (point.payload or {}).items() if k != "text"}
                documents.append(SearchResult(
                    text=text,
                    metadata=meta,
                    score=0.0,
                    index=index,
                    retrieval_source="sparse",
                ))
                index += 1

            if next_offset is None:
                break
            offset = next_offset

        return documents

    def bump_version(self) -> None:
        self._document_version += 1

    def delete_collection(self) -> None:
        """
        删除整个集合（包含所有向量和 payload）

        用于测试清理或重置知识库，操作不可逆，谨慎调用。
        """
        self._client.delete_collection(self._collection)
        self.bump_version()
        logger.info(f"Deleted Qdrant collection: {self._collection}")
