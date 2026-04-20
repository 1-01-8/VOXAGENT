# VoxCareAgent 面试速查（Agent / LLM / RAG 实现要点）

> 只讲这个项目里**真实存在**的实现。每条都能在代码里找到对应文件。
> 用于面试前 10 分钟 quick review。

---

## 0. 一句话总定位（开场白）

> "这是一个**语音驱动的 RAG + ReAct Agent 智能客服系统**。技术栈是 FastAPI + asyncio 异步全链路，
> LLM 用本地 Ollama qwen2.5:7b，Embedding 用 nomic-embed-text，向量库 FAISS IndexFlatIP 内存检索，
> ASR/TTS 用 SiliconFlow 云端的 SenseVoiceSmall + CosyVoice2，通过 WebSocket 流式推送 PCM 到浏览器
> AudioContext 播放。核心亮点是双 Agent 预取架构把语音 RAG 的端到端延迟从秒级压到 ~800ms。"

---

## 1. RAG 部分

### 1.1 用到的技术与组件

| 角色 | 选型 | 位置 |
|------|------|------|
| Embedding | **Ollama nomic-embed-text** (768d) / 可切 OpenAI text-embedding-3-small (1536d) | `retrieval/embeddings.py` |
| Vector Store | **FAISS `IndexFlatIP`**（内积 + L2 归一化 = 余弦相似度）；备选 Qdrant | `retrieval/vector_store.py` / `qdrant_store.py` |
| Chunking | 512 字符 + 50 字符 overlap，按字符滑动切 | `retrieval/document_loader.py::chunk_text` |
| 语义缓存 | FAISS IndexFlatIP + TTL + LRU | `core/semantic_cache.py` |
| 双 Agent | SlowThinker (后台预测 + 预取) + FastTalker (前台查缓存) | `core/slow_thinker.py` / `fast_talker.py` |
| 事件总线 | `ConversationStream` = `list[asyncio.Queue]` 发布订阅 | `core/conversation_stream.py` |

### 1.2 RAG 检索链路（面试高频问）

```
query
 ├─ [1] embed(query)                      ~30ms  (Ollama 本地)
 ├─ [2] SemanticCache.get(emb, k=top_k)   <1ms   (FAISS 内存命中)  ──┐
 │     ↑ 阈值 0.40 余弦                                               │ 命中跳到[5]
 ├─ [3] FAISSVectorStore.search(emb, k=3) ~1-3ms (内存索引)          │
 ├─ [4] SemanticCache.put(emb, top1)                                 │
 └─ [5] LLM.generate(prompt + retrieved_docs)  ~200-400ms TTFT  ←────┘
```

**关键算法点**：
- **为什么 IndexFlatIP 不用 IndexFlatL2？** 内积 + L2 归一化 = 余弦相似度；只看方向不看长度，适合文本语义。
- **为什么 overlap=50？** 句子被切散时（"……售价是 | 2999 元……"），重叠保证任一 chunk 都能独立回答。
- **SemanticCache 的 key 是什么？** **文档块自身的向量**（不是查询向量），所以 "多少钱" / "什么价格" 两个问法能命中同一条缓存。
- **双 Agent 怎么协作？** SlowThinker 订阅 `USER_UTTERANCE` 事件 → LLM 预测下一问 → 预检索 → 写入 SemanticCache。FastTalker 主路径先查缓存，命中 <1ms 直接返回。
- **为什么双 Agent 不抢资源？** SlowThinker 通过 `asyncio.create_task` 在事件循环里并行，I/O 等待时自然让出，**0ms 阻塞主路径**。

### 1.3 我写的 RAG 性能测试工具

`tests/bench_rag_match.py`，分四段测：embedding / cache lookup / vector_store search / total cold-warm，
输出 p50/p95/avg/max 表格 + 缓存命中率 + warm-on-hit 加速比。命令：

```bash
python tests/bench_rag_match.py --rounds 3 --top-k 5
```

---

## 2. LLM 部分

### 2.1 多 provider 抽象层

`llm/base.py` 定义 `LLMProvider` 抽象类（`generate()` / `stream()`），`create_llm(config)` 工厂方法按 `config.llm_provider` 分发。
实现了 5 个后端，都支持 async 流式：

| Provider | 文件 | 场景 |
|----------|------|------|
| **Ollama** ⭐ | `llm/ollama_provider.py` | 当前默认，本地 qwen2.5:7b |
| OpenAI | `llm/openai_provider.py` | 兼容 Azure OpenAI / Salesforce Gateway |
| Anthropic | `llm/anthropic_provider.py` | Claude |
| Gemini | `llm/gemini_provider.py` | Google |
| SiliconFlow | `llm/siliconflow_provider.py` | 云端 qwen/DeepSeek |

### 2.2 LLM 在项目里的 6 个职责

| 职责 | 调用位置 | Prompt 关键词 |
|------|---------|--------------|
| 1) 意图分类（task/knowledge/chitchat） | `dialogue/intent_router.py` | "请只返回一个词" |
| 2) ReAct 推理 | `agent/react_agent.py` | Thought/Action/Observation/Final Answer |
| 3) 预测下一问题 | `core/slow_thinker.py` | "预测用户下一个可能的问题" |
| 4) 多轮摘要压缩 | `dialogue/memory_manager.py` | "压缩为 200 字以内摘要" |
| 5) 知识答 / 闲聊答（流式） | `web/server.py::_produce_from_stream` | "基于上下文回答" / "友好回复闲聊" |
| 6) 语义一致性检查（可选） | `utils/auto_qa.py::_check_consistency` | Embedding 余弦对比 |

### 2.3 降级策略

`web/server.py::_try_load_llm` 加载失败返回 `_EchoLLM`（只回显提示信息），前端不会崩。

---

## 3. Agent 部分（ReAct）

### 3.1 为什么选 ReAct 不选 Function Calling

| 维度 | ReAct (本项目) | OpenAI Function Calling |
|------|---------------|------------------------|
| LLM 无关 | ✅ Ollama / Claude / Gemini 都能用 | ❌ 仅 OpenAI 系 |
| 推理透明 | ✅ Thought 可审计 | ❌ 黑盒 |
| 本地部署 | ✅ 纯文本协议 | ❌ 需专属 API |
| 解析可靠 | ⚠️ 文本解析易错 → 用括号平衡法兜底 | ✅ 原生 JSON Schema |

### 3.2 ReAct 循环（`agent/react_agent.py`）

```python
for iteration in range(max_iterations=10):
    if len(scratchpad) > 6000:        # 防爆 context
        scratchpad = header + "...(省略)..." + tail

    response = await llm.generate(system_prompt + scratchpad)
    scratchpad += response

    # ─ 出口 A: Final Answer (< 6 字视为残缺，继续循环) ─
    final = _extract_final_answer(response)
    if final: return final

    # ─ 解析工具调用 ─
    name, args = _parse_action(response)   # 括号平衡法，支持嵌套 JSON

    # ─ 权限 ─
    if not await permission_guard.check(...): continue

    # ─ 执行（timeout 3s + retry 1 次）─
    result = await execute_tool_with_retry(tool, args)
    scratchpad += f"Observation: {result}\n"

# ─ 出口 B/C: 失败>=3 / 迭代超限 → TRANSFER_REQUEST 事件 → 转人工
```

### 3.3 四级权限系统（`agent/permission_guard.py`）

```
L1 只读   → 直接放行             例: QueryOrder / QueryInventory
L2 写入   → 发 CONFIRM_REQUIRED  → asyncio.Future 等前端 15s 确认
            例: UpdateAddress / CancelOrder
L3 财务   → L2 + 身份二次验证    例: ApplyRefund
L4 管理   → 直接拒绝（Agent 不得执行）
```

**关键实现：`_pending_confirms: dict[request_id, asyncio.Future]`** — 按 `request_id` 隔离每个请求，避免并发竞态。前端回传 `request_id` 精确匹配 `future.set_result(bool)`，`asyncio.wait_for(future, timeout=15)` 超时自动拒绝。**事件驱动零轮询**。

### 3.4 7 个业务工具（`agent/tools/`）

| Tool | Level | 参数 | 文件 |
|------|:-:|------|------|
| QueryOrder / QueryInventory / GetCustomerInfo / CheckPromotion | L1 | 对应 id/name | `query_tools.py` |
| UpdateAddress / CancelOrder | L2 | order_id + new_value | `write_tools.py` |
| ApplyRefund | L3 | order_id + amount | `finance_tools.py` |

所有工具继承 `base_tool.BaseTool`，抽象层完整（参数校验 / Prompt 描述生成 / 权限标注）；`execute()` 目前返回 Mock，接入真实 API 只需改一个方法。

### 3.5 三路意图路由（`dialogue/intent_router.py`）

```python
classify(utterance):
    # ①最高：转人工关键词 → 标记 transfer_requested，路由到 CHITCHAT 先安抚
    # ②TASK 关键词 (查订单/退款/取消/修改地址...)       ← 零延迟
    # ③KNOWLEDGE 关键词 (多少钱/报价/套餐/保修/对比...) ← 零延迟
    # ④LLM 三路兜底 (prompt 要求只返回一个词)
    # LLM 失败时默认 KNOWLEDGE（最安全）
```

### 3.6 转人工策略 5 规则（`dialogue/transfer_policy.py`）

1. 持续愤怒轮次 ≥ 阈值（默认 2）
2. Agent 连续失败 ≥ 3 次
3. VIP (level≥3) + 情绪 ANGRY/SAD
4. 高风险关键词（法律 / 律师 / 起诉 / 消协）
5. 用户主动要求（由 IntentRouter 前置检测）

---

## 4. 异步编程关键模式（面试肯定问）

| 模式 | 用途 | 代码位置 |
|------|------|---------|
| `asyncio.create_task` | SlowThinker 后台并行不阻塞主路径 | `memory_router.py::start()` |
| `run_in_executor` | CPU 密集 (FunASR 推理) 丢线程池 | `voice/sensevoice_stt.py` |
| `asyncio.Future` | 等用户确认，事件驱动零轮询 | `permission_guard.py` |
| `asyncio.Queue` | 同步生成器 ↔ 异步迭代器 桥接 | `voice/cosyvoice.py::synthesize_stream` |
| `asyncio.Lock` | SemanticCache 并发读写 | `core/semantic_cache.py` |
| `get_running_loop` 替代 `get_event_loop` | 3.10+ 语义正确 | 5 个文件 |
| httpx AsyncClient keep-alive 池 | SiliconFlow API 连接复用 | `voice/siliconflow_*.py` |
| WebSocket HTTP chunked → audio_chunk | TTS 首帧 < 1s | `web/server.py::_tts_consumer` |

---

## 5. 语音链路（v1.0.x 迁移后）

### STT (`voice/siliconflow_stt.py`)
```
PCM bytes → _pcm_to_wav() → multipart/form-data → POST /v1/audio/transcriptions
  model=FunAudioLLM/SenseVoiceSmall → 返回 {text, emotion?}
```

### TTS (`voice/siliconflow_tts.py`)
```
text → POST /v1/audio/speech  (stream=true, format=pcm, sample_rate=24000)
     → async for chunk in response.aiter_bytes(4800):   # 4800 ≈ 100ms @ 24kHz/16bit
         yield chunk            → ws.send audio_chunk → AudioContext.connect 播放
```

### 前端（`web/static/index.html`）
用 `AudioContext` @ 24kHz，维护 `streamNextStart` 拼接每块 PCM，保证连续无爆音。

---

## 6. 数据流全景图（10 秒白板画图版）

```
Mic → WS → STT(SF) → Emotion → IntentRouter ─┬─ TASK:    ReActAgent → Tools(+Guard)
                                              ├─ KNOWL:   LLM stream + FAISS + Cache
                                              └─ CHITCHAT:LLM stream only
                                                    ↓
                                              AutoQA 质检
                                                    ↓
                                         TTS(SF chunked) → WS → AudioCtx
                                                    ↓
                                         Memory add_turn (异步压缩)
                                                    ↓
                                         SessionLogger 异步 JSONL
```

---

## 7. 容易被追问的 7 个"陷阱问题"

1. **RAG 阈值 0.40 为什么这么低？** 因为 query-to-document 的余弦比 query-to-query 低，doc 更"冗长"；实际调出来的甜点是 0.35-0.50。
2. **FAISS 没有持久化的问题？** `vector_store.save/load` 用 `faiss.write_index` + metadata.json；Qdrant 版本是为了多实例共享。
3. **qwen2.5:7b 的 ReAct 稳定性？** 不稳。项目里做了 3 层兜底：(a) 关键词优先走 RAG 不进 Agent；(b) Final Answer < 6 字视为残缺继续循环；(c) 迭代 10 轮仍无果则转人工。
4. **TTS 怎么做到 800ms 首帧？** SF `stream=true` + HTTP chunked + 按句切片（MIN_FLUSH_CHARS=6 + 标点触发 flush），第一句 6-10 字合成 ~500ms + 网络 ~300ms。
5. **多轮记忆 token 怎么控？** 短期保 10 轮原文，溢出时 LLM 压缩为 ≤200 字摘要，prompt 里只注入"摘要 + 最近原文"。
6. **并发会话如何隔离？** WebSocket 连接内实例化独立 `SessionContext` + `MemoryManager`，只共享 LLM/Embedding/VectorStore/Cache 只读组件。
7. **如果 Ollama 挂了？** `_try_load_llm` 降级 `EchoLLM`，前端收到 "Echo 演示模式" 仍可交互；RAG 路径全废，但 WebSocket 不断。

---

## 8. 数字记忆表（背这几个数就够了）

| 指标 | 数字 | 出处 |
|------|:----:|------|
| Embedding 维度 | 768 (Ollama) / 1536 (OpenAI) | config.py |
| Chunk 大小 | 512 字符 + 50 overlap | config.py |
| SemanticCache 容量 | 2000 条 | config.py |
| SemanticCache TTL | 300s | config.py |
| SemanticCache 阈值 | 0.40 余弦 | config.py |
| ReAct 最大迭代 | 10 | config.py |
| ReAct scratchpad 上限 | 6000 字符 | react_agent.py |
| Tool 超时 | 3s + retry 1 次 | config.py |
| Confirm 超时 | 15s | permission_guard.py |
| 短期记忆窗口 | 10 轮 | config.py |
| Agent 连续失败阈值 | 3 | transfer_policy.py |
| 持续愤怒阈值 | 2 轮 | transfer_policy.py |
| TTS 切片最小字数 | 6 | server.py |
| TTS chunk 大小 | 4800 B (~100ms @ 24kHz/16bit) | siliconflow_tts.py |
| TTS 采样率 | 24kHz | config.py |
| STT 采样率 | 16kHz | config.py |
| 单元测试数 | 137 全 PASS | tests/ |

---

*如果面试官只给 30 秒自我介绍：背第 0 节。如果给 2 分钟：第 0 节 + 第 6 节白板图。如果给 10 分钟：按 RAG → LLM → Agent → 异步 → 语音 的顺序讲。*
