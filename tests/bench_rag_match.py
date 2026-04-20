"""RAG 匹配速度基准测试工具

测量以下四个阶段的耗时，定位端到端 RAG 的性能瓶颈：

    Query
      │
      ├─ [1] Embedding     ── 文本 → 向量（通常 20-50ms，本地 Ollama / 远程 API）
      │
      ├─ [2] SemanticCache ── 先查缓存（内存 FAISS，< 1ms，命中则跳过 3）
      │
      ├─ [3] VectorStore   ── FAISS 磁盘索引检索 top-k（几 ms）
      │
      └─ [4] Total         ── 端到端墙钟时间

用法：
    # 1) 先灌库（若已有 data/faiss_index 可跳过）
    python examples/ingest_documents.py knowledge_base/ \\
        --provider ollama --embedding-provider ollama

    # 2) 跑基准（默认 20 个中文客服问法）
    python tests/bench_rag_match.py

    # 3) 自定义参数
    python tests/bench_rag_match.py --top-k 5 --rounds 3 \\
        --queries "你们的价格怎么样" "怎么退款" "SLA 是多少"

输出示例：
    ┌──────────────────────┬─────────┬─────────┬─────────┬─────────┐
    │ stage                │  p50 ms │  p95 ms │  avg ms │  max ms │
    ├──────────────────────┼─────────┼─────────┼─────────┼─────────┤
    │ embedding            │   28.4  │   41.2  │   29.8  │   45.1  │
    │ semantic_cache_miss  │    0.3  │    0.5  │    0.3  │    0.7  │
    │ vector_store_top5    │    1.2  │    2.8  │    1.5  │    3.1  │
    │ total (cold)         │   30.1  │   43.4  │   31.8  │   47.2  │
    │ total (warm cache)   │    0.4  │    0.6  │    0.4  │    0.8  │
    └──────────────────────┴─────────┴─────────┴─────────┴─────────┘

    Cache hit rate: 95.0%  ·  Speedup on hit: 79.5x
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from voice_optimized_rag.config import VORConfig
from voice_optimized_rag.core.semantic_cache import SemanticCache
from voice_optimized_rag.retrieval.embeddings import create_embedding_provider
from voice_optimized_rag.retrieval.vector_store import FAISSVectorStore


# ── 默认查询集（覆盖三路意图：知识 / 任务 / 闲聊） ──
DEFAULT_QUERIES: list[str] = [
    "你们的产品多少钱",
    "有什么优惠活动",
    "怎么申请退款",
    "保修期是多久",
    "支持哪些支付方式",
    "物流什么时候到",
    "企业版和个人版有什么区别",
    "API 调用限制是多少",
    "数据安全怎么保障",
    "怎么对接 CRM",
    "报价方案给我一份",
    "月费大概是多少",
    "能试用吗",
    "支持私有化部署吗",
    "客服工作时间是几点",
    "怎么升级套餐",
    "取消订单需要多久",
    "发票怎么开",
    "有没有案例参考",
    "新手怎么上手",
]


# ─── 统计工具 ───

def _stats(samples_ms: Sequence[float]) -> dict[str, float]:
    """返回 p50 / p95 / avg / max 毫秒。"""
    if not samples_ms:
        return {"p50": 0.0, "p95": 0.0, "avg": 0.0, "max": 0.0}
    sorted_s = sorted(samples_ms)
    n = len(sorted_s)
    return {
        "p50": sorted_s[n // 2],
        "p95": sorted_s[min(n - 1, int(n * 0.95))],
        "avg": statistics.fmean(sorted_s),
        "max": sorted_s[-1],
    }


def _print_table(rows: list[tuple[str, dict[str, float]]]) -> None:
    """以固定宽度表格打印基准结果。"""
    print("\n┌──────────────────────┬─────────┬─────────┬─────────┬─────────┐")
    print("│ stage                │  p50 ms │  p95 ms │  avg ms │  max ms │")
    print("├──────────────────────┼─────────┼─────────┼─────────┼─────────┤")
    for name, s in rows:
        print(
            f"│ {name:<20} │ {s['p50']:>7.2f} │ {s['p95']:>7.2f} │ "
            f"{s['avg']:>7.2f} │ {s['max']:>7.2f} │"
        )
    print("└──────────────────────┴─────────┴─────────┴─────────┴─────────┘")


# ─── 主基准逻辑 ───

async def run_benchmark(
    queries: Sequence[str],
    top_k: int,
    rounds: int,
    cache_threshold: float,
    index_path: Path,
) -> None:
    """跑完整基准：embedding → cache → vector_store → total。"""

    print(f"[bench] loading config + providers ...")
    config = VORConfig()
    embedder = create_embedding_provider(config)

    print(f"[bench] loading FAISS index from {index_path} (dim={config.embedding_dimension}) ...")
    store = FAISSVectorStore(
        dimension=config.embedding_dimension,
        index_path=index_path,
    )
    if store.size == 0:
        print(
            f"⚠️  向量库为空。请先运行:\n"
            f"    python examples/ingest_documents.py knowledge_base/ "
            f"--provider ollama --embedding-provider ollama"
        )
        return

    cache = SemanticCache(
        dimension=config.embedding_dimension,
        max_size=config.cache_max_size,
        default_ttl=config.cache_ttl_seconds,
        similarity_threshold=cache_threshold,
    )

    # 每个 bucket 的采样点
    t_embed_ms: list[float] = []
    t_cache_lookup_ms: list[float] = []
    t_vstore_ms: list[float] = []
    t_total_cold_ms: list[float] = []
    t_total_warm_ms: list[float] = []

    cache_hits = 0
    cache_misses = 0

    print(
        f"[bench] running {rounds} round(s) × {len(queries)} queries "
        f"· top_k={top_k} · cache_threshold={cache_threshold}\n"
    )

    for round_idx in range(rounds):
        for q in queries:
            # ── stage 1: Embedding ──
            t = time.perf_counter()
            q_emb = await embedder.embed_single(q)
            t_embed_ms.append((time.perf_counter() - t) * 1000)

            q_emb_2d = q_emb.reshape(1, -1).astype(np.float32).copy()

            # ── stage 2: SemanticCache 查询 ──
            t = time.perf_counter()
            hit = await cache.get(q_emb_2d[0], top_k=top_k)
            cache_ms = (time.perf_counter() - t) * 1000
            t_cache_lookup_ms.append(cache_ms)

            t_total = time.perf_counter()

            if hit:
                # 命中缓存 → 端到端仅包含 embedding + cache 查询
                cache_hits += 1
                total_ms = (time.perf_counter() - t_total) * 1000 + \
                    t_embed_ms[-1] + cache_ms
                t_total_warm_ms.append(total_ms)
                continue

            cache_misses += 1

            # ── stage 3: VectorStore 检索 ──
            t = time.perf_counter()
            results = store.search(q_emb, top_k=top_k)
            vstore_ms = (time.perf_counter() - t) * 1000
            t_vstore_ms.append(vstore_ms)

            # ── stage 4: Total (cold) = embed + cache + vstore ──
            total_cold_ms = t_embed_ms[-1] + cache_ms + vstore_ms
            t_total_cold_ms.append(total_cold_ms)

            # 回填缓存：用 top-1 结果写入（后续同义问法可命中）
            if results:
                top = results[0]
                await cache.put(
                    query_embedding=q_emb_2d[0],
                    text=top.text,
                    metadata=top.metadata,
                    relevance_score=float(top.score),
                )

    # ─── 输出 ───
    rows: list[tuple[str, dict[str, float]]] = [
        ("embedding",           _stats(t_embed_ms)),
        ("semantic_cache_look", _stats(t_cache_lookup_ms)),
        (f"vector_store_top{top_k}", _stats(t_vstore_ms)),
        ("total (cold)",        _stats(t_total_cold_ms)),
        ("total (warm cache)",  _stats(t_total_warm_ms)),
    ]
    _print_table(rows)

    total_queries = cache_hits + cache_misses
    hit_rate = (cache_hits / total_queries * 100) if total_queries else 0.0
    cold_p50 = _stats(t_total_cold_ms)["p50"]
    warm_p50 = _stats(t_total_warm_ms)["p50"]
    speedup = (cold_p50 / warm_p50) if warm_p50 > 0 else 0.0

    print(
        f"\nCache hit rate: {hit_rate:.1f}%  "
        f"({cache_hits}/{total_queries})"
        f"  ·  Speedup on hit: {speedup:.1f}x"
        f"  ·  VectorStore size: {store.size} chunks"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 端到端匹配耗时基准（embedding + cache + vector_store）"
    )
    parser.add_argument(
        "--queries", nargs="+", default=None,
        help="自定义查询列表（空则用内置 20 条中文客服问法）",
    )
    parser.add_argument(
        "--top-k", type=int, default=3,
        help="VectorStore / SemanticCache 每次检索返回条数",
    )
    parser.add_argument(
        "--rounds", type=int, default=2,
        help="每条查询重跑几轮（第 2 轮起能观察到缓存命中）",
    )
    parser.add_argument(
        "--cache-threshold", type=float, default=0.40,
        help="SemanticCache 命中阈值（query-doc 余弦），默认对齐 VORConfig",
    )
    parser.add_argument(
        "--index-path", type=Path,
        default=Path("data/faiss_index"),
        help="FAISS 索引目录",
    )
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    asyncio.run(run_benchmark(
        queries=queries,
        top_k=args.top_k,
        rounds=args.rounds,
        cache_threshold=args.cache_threshold,
        index_path=args.index_path,
    ))


if __name__ == "__main__":
    main()
