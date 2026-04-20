# VoxCareAgent 项目总览（Project Overview）

> **版本**: v1.0.1 · 合并版
> **最后更新**: 2026-04-20
> **定位**: 替代分散在仓库根目录的 7+ 份历史文档，作为**唯一真相源 (SSOT)**
>
> 历史文档（V1 白皮书 / V2 方案 / 4.20 部署任务 / 启动指南 / 环境说明 / 项目书）已归并到本文档，
> 原文件已删除。保留的姊妹文档：
> - [README.md](README.md) — 英文项目简介
> - [VoiceAgentRAG_技术架构说明书.md](VoiceAgentRAG_技术架构说明书.md) — 面试深度版（架构/算法细节）
> - [V1.0.0_全链路低延迟Debug文档.md](V1.0.0_全链路低延迟Debug文档.md) — 语音迁移 SiliconFlow 的专项 Debug 记录
> - [RUNTIME_GUIDE.md](RUNTIME_GUIDE.md) — 通用运行指南
> - [SECURITY_REMINDER.md](SECURITY_REMINDER.md) — 密钥泄漏风险清单
> - [INTERVIEW_BRIEF.md](INTERVIEW_BRIEF.md) — 面试速查（Agent / LLM / RAG 要点）

---

## 一、系统定位

**VoxCareAgent**（原名 VoiceAgentRAG，Salesforce Research 开源底座 + 客服场景业务化改造）是一个
**语音驱动的 RAG + ReAct Agent 智能客服系统**，完成：

    🎤 语音 → ASR+情绪 → 意图路由 → RAG / ReAct 工具 → 质检 → TTS 流式回放 → 🔊

### 关键指标（v1.0.1 实测）

| 指标 | 目标 | 实测 | 说明 |
|------|-----|------|------|
| TTS 首帧 (TTFB) | < 1s | ~800ms | SiliconFlow 云端 CosyVoice2 + HTTP chunked |
| RAG 端到端 (cold) | < 100ms | 30-50ms | Ollama `nomic-embed` 本地 + FAISS 内存 |
| RAG 端到端 (warm) | < 5ms | < 1ms | SemanticCache FAISS 内积命中 |
| LLM TTFT | < 500ms | 200-400ms | Ollama qwen2.5:7b 本地 |
| 并发会话 | ≥ 100 | 全链路 async | 单线程事件循环 |

---

## 二、当前部署架构（v1.0.1 · 混合部署）

```
┌─── 本地 (GPU 服务器) ──────────────────────────┐
│                                                │
│  Ollama :11434                                 │
│  ├── qwen2.5:7b           LLM                  │
│  └── nomic-embed-text     Embedding (768d)     │
│                                                │
│  FastAPI + Uvicorn :8000                       │
│  ├── WebSocket /ws/chat   主流水线              │
│  ├── FAISS IndexFlatIP    向量库 (内存)         │
│  └── SemanticCache        语义缓存 (FAISS)     │
│                                                │
└────────────────────────────────────────────────┘
                     │ HTTPS
                     ▼
┌─── SiliconFlow 云端 ───────────────────────────┐
│  POST /v1/audio/transcriptions                 │
│    model = FunAudioLLM/SenseVoiceSmall (ASR)   │
│  POST /v1/audio/speech                         │
│    model = FunAudioLLM/CosyVoice2-0.5B (TTS)   │
│    voice = alex · stream = true · format = pcm │
└────────────────────────────────────────────────┘
```

### 迁移决策记录（从 v0.x 纯本地 → v1.0.x 混合）

| 组件 | v0.x | v1.0.x | 原因 |
|------|------|--------|------|
| LLM | Ollama 本地 | Ollama 本地 | TTFT 200-400ms，正常 |
| Embed | Ollama 本地 | Ollama 本地 | 单次 ~30ms，正常 |
| ASR | 本地 SenseVoice (funasr) | **SiliconFlow 云端** | 降低 GPU 占用；本地常驻 2GB 显存不划算 |
| TTS | 本地 CosyVoice2 | **SiliconFlow 云端** | 本地 RTF 6-10 (11 字需 14-20s)；cuDNN/TRT 环境冲突严重 |

---

## 三、目录结构

```
VoxCareAgent/
├── voice_optimized_rag/          # 主包（装作 pip -e .）
│   ├── config.py                 # VORConfig (Pydantic Settings, VOR_ 前缀)
│   ├── core/                     # 双代理 + 事件总线 + 缓存
│   │   ├── conversation_stream.py  # ConversationStream 事件总线
│   │   ├── fast_talker.py          # 前台 Agent (主路径)
│   │   ├── slow_thinker.py         # 后台预取 Agent
│   │   ├── semantic_cache.py       # FAISS 语义缓存 + TTL + LRU
│   │   └── memory_router.py        # 总调度器
│   ├── retrieval/                # RAG 组件
│   │   ├── embeddings.py           # OpenAI / Ollama / Local providers
│   │   ├── vector_store.py         # FAISSVectorStore (IndexFlatIP)
│   │   ├── qdrant_store.py         # Qdrant 替代实现
│   │   ├── document_loader.py      # 分块 (512 chars + 50 overlap)
│   │   └── kb_manager.py           # 热更新接口
│   ├── llm/                      # LLM 抽象 + 5 后端
│   │   ├── base.py                 # LLMProvider 接口 + create_llm 工厂
│   │   ├── ollama_provider.py      # 当前默认
│   │   ├── openai_provider.py
│   │   ├── anthropic_provider.py
│   │   ├── gemini_provider.py
│   │   └── siliconflow_provider.py
│   ├── voice/                    # 语音模块
│   │   ├── stt.py / tts.py         # 工厂
│   │   ├── sensevoice_stt.py       # 本地 FunASR（降级路径）
│   │   ├── cosyvoice.py            # 本地 CosyVoice2（降级路径）
│   │   ├── siliconflow_stt.py      # ⭐ 云端 ASR（当前默认）
│   │   └── siliconflow_tts.py      # ⭐ 云端 TTS 流式（当前默认）
│   ├── dialogue/                 # 对话管理
│   │   ├── session.py              # SessionContext (情绪/意图/多轮)
│   │   ├── intent_router.py        # 关键词 + LLM 三路分类
│   │   ├── emotion_detector.py     # 标签映射 + 状态机
│   │   ├── memory_manager.py       # 短期原文 + 中期摘要
│   │   └── transfer_policy.py      # 5 规则转人工
│   ├── agent/                    # ReAct Agent
│   │   ├── base_tool.py            # BaseTool 抽象类
│   │   ├── react_agent.py          # ReAct 循环（Thought/Action/Obs）
│   │   ├── permission_guard.py     # 4 级权限 + Future 确认流
│   │   └── tools/                  # 7 个业务工具
│   │       ├── query_tools.py      # L1: 订单/库存/客户/促销
│   │       ├── write_tools.py      # L2: 地址/取消
│   │       └── finance_tools.py    # L3: 退款
│   └── utils/                    # 工具集
│       ├── auto_qa.py              # 质检 (敏感词 / 禁止模式 / 语义一致性)
│       ├── session_logger.py       # 异步 JSONL 审计
│       └── metrics.py              # 指标采集
├── web/                          # FastAPI + WebSocket
│   ├── server.py                 # 主入口：lifespan / /ws/chat
│   └── static/index.html         # 前端（AudioContext 24kHz 流式播放）
├── examples/
│   ├── ingest_documents.py       # 灌库脚本
│   ├── benchmark.py              # 端到端延迟基准
│   ├── cli_demo.py / voice_demo.py / customer_service_demo.py
├── tests/
│   ├── test_*.py                 # 137 unit tests, all passing
│   └── bench_rag_match.py        # ⭐ RAG 匹配速度测试工具（新增）
├── knowledge_base/               # 12 份纯文本 FAQ
├── data/faiss_index/             # 序列化的 FAISS 索引
├── logs/sessions/                # JSONL 会话审计日志
├── CosyVoice/                    # 官方仓库（备用，本地 TTS）
└── .env / .env.example           # 配置（见下节）
```

---

## 四、一键启动

### 4.1 环境准备（一次性）

```bash
# Conda 统一环境
conda create -n voxcare python=3.10 -y
conda activate voxcare

# 安装依赖（pyproject.toml 定义）
pip install -e .

# 本地 Ollama（LLM + Embedding）
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 4.2 配置 .env

```bash
# ── 本地 ──
VOR_LLM_PROVIDER=ollama
VOR_LLM_MODEL=qwen2.5:7b
VOR_LLM_BASE_URL=http://localhost:11434
VOR_EMBEDDING_PROVIDER=ollama
VOR_EMBEDDING_MODEL=nomic-embed-text
VOR_EMBEDDING_DIMENSION=768

# ── 云端语音 ──
VOR_STT_PROVIDER=siliconflow
VOR_TTS_PROVIDER=siliconflow
VOR_SILICONFLOW_API_KEY=sk-xxxxxxxxxxxxxxxx   # ⚠️ 务必自己填
VOR_SILICONFLOW_TTS_VOICE=alex
```

### 4.3 启动

```bash
conda activate voxcare

# 1) 灌库（首次或 knowledge_base/ 有变更时）
python examples/ingest_documents.py knowledge_base/ \
    --provider ollama --embedding-provider ollama

# 2) 启 Web 服务
uvicorn web.server:app --host 0.0.0.0 --port 8000

# 3) 浏览器访问 http://<server-ip>:8000
```

### 4.4 基准测试

```bash
# RAG 匹配耗时基准（新工具）
python tests/bench_rag_match.py --rounds 3 --top-k 5

# 端到端 TTS 延迟基准
python bench_tts.py
```

---

## 五、核心业务流水线（`web/server.py`）

```
WebSocket 收到 {type:"audio",data:<b64 PCM>} 或 {type:"text",text:"..."}
    │
    ├─[1] STT：SiliconFlowSTT.transcribe_with_emotion()
    │       → {text, emotion, event}
    │
    ├─[2] EmotionDetector.update()：情绪映射 + 发布 EMOTION_CHANGE
    │
    ├─[3] IntentRouter.classify()：
    │       ├ 转人工关键词 → 返回 CHITCHAT + 设 transfer_requested
    │       ├ TASK 关键词 (查订单/退款/取消)
    │       ├ KNOWLEDGE 关键词 (报价/价格/套餐/保修...)  ← v1.0.1 扩充
    │       └ LLM 三路兜底
    │
    ├─[4] TransferPolicy.evaluate()：5 规则 → 命中则直接回"正在转人工"
    │
    ├─[5] MemoryManager.add_turn(user, ...) 异步
    │
    ├─[6] 按意图分流：
    │   ├─ TASK     → ReactAgent.execute() 10 轮循环，Final Answer < 6 字兜底
    │   ├─ KNOWLEDGE → _produce_from_stream：LLM 流式 token → 按标点切片送 tts_queue
    │   └─ CHITCHAT  → 同 KNOWLEDGE 走 LLM 直答
    │
    ├─[7] _tts_consumer (后台任务)：从 tts_queue 取片段
    │     ├─ 调 SiliconFlowTTS.synthesize_http_stream()
    │     └─ 每块 4800B PCM → ws.send_json({type:"audio_chunk", audio:b64, sample_rate:24000})
    │
    ├─[8] AutoQA.check()：敏感词警告 / 禁止模式替换为兜底回复
    │
    ├─[9] SessionLogger.log_turn() 异步 JSONL 落盘
    │
    ├─[10] ws.send_json({type:"reply", text, emotion, intent, timing_ms})
    │
    └─[11] await consumer_task：等 tts_queue 清空 + audio_end
```

> **v1.0.1 关键修复**
> - 删除了 `_tts_stream_and_send` 重复合成（导致 "多重声效" 叠音）
> - IntentRouter `KNOWLEDGE_KEYWORDS` 追加 `报价/费用/套餐/方案/收费/介绍`
> - ReactAgent `_extract_final_answer` < 6 字视为残缺格式，防止 "为了" 截断输出
> - 移除本地 TTS `_ensure_model` 预热在非本地 provider 下的无效调用（`hasattr` 守卫）

---

## 六、关键设计要点

> 详细论述见 [VoiceAgentRAG_技术架构说明书.md](VoiceAgentRAG_技术架构说明书.md)，本节只列速查纲要。

| 主题 | 要点 |
|------|------|
| **双代理** | SlowThinker (asyncio.create_task) 订阅事件预测下一问题 → 预检索 → SemanticCache |
| **事件总线** | ConversationStream = list[asyncio.Queue]，9 种事件类型，背压自动限流 |
| **语义缓存** | FAISS IndexFlatIP + L2归一化 = 余弦相似度；TTL 过期 + LRU 淘汰 |
| **RAG 分块** | chunk_size=512，overlap=50（保切割边界信息完整性） |
| **ReAct** | 10 轮循环上限；scratchpad > 6000 字截断中间；括号平衡法解析嵌套 JSON |
| **4 级权限** | L1 只读→直接；L2 写入→`asyncio.Future` 等前端确认 15s；L3 财务→二次验证；L4 禁用 |
| **三级记忆** | 短期原文 10 轮 + 中期 LLM 压缩摘要 + 长期 JSONL |
| **转人工** | 5 规则：持续愤怒/Agent 失败/VIP+不满/高风险词/主动要求 |
| **质检** | 敏感词仅警告 + 禁止模式替换兜底 + 可选语义一致性 |
| **降级** | LLM → EchoLLM；STT → 前端文字输入；TTS → Web Speech API |

---

## 七、已知限制（不影响上线，Roadmap）

| # | 项目 | 说明 | 计划 |
|---|------|------|------|
| 1 | 7 个业务工具返回 Mock 数据 | BaseTool 抽象层已完成，execute() 硬编码 | 对接真实 CRM/订单 API |
| 2 | ConversationStream 无 Queue 上限 | 慢消费者潜在内存泄漏 | 增加 maxsize + 丢弃策略 |
| 3 | KBManager 访问私有属性 | 直接操作 `_metadata`/`_texts` | 暴露公开 upsert/delete API |
| 4 | 敏感词硬编码 | "保证"/"一定" 误报率高 | 移到 config + 上下文感知 |
| 5 | 无流式 STT | 需要用户发完整音频段 | 对接 SiliconFlow WebSocket 版（待其上线） |
| 6 | 前端无 Jitter Buffer | 跨洋网络抖动时偶尔断音 | v1.3 增加 200ms 缓冲池 |

---

## 八、故障排查速查

| 现象 | 根因 | 定位 | 修复 |
|------|-----|------|------|
| 回复只有 "为了" / "好的" | qwen2.5:7b 在 ReAct prompt 下输出残缺 Final Answer | 查 IntentRouter 是否误判 TASK | `KNOWLEDGE_KEYWORDS` 加词 **或** 升级 `qwen2.5:14b` |
| 听到回声 / 多重声效 | TTS 管线重复合成 | server.py 检查是否有两处调 `audio_chunk` 推送 | 已在 v1.0.1 删除冗余 `_tts_stream_and_send` |
| TTS 首帧 > 3s | SF API 无流式 / `stream=false` | 查 siliconflow_tts.py 是否用 `synthesize_http_stream` | 确认 `.env` 未覆盖 stream 参数 |
| 启动报 `cuDNN version incompatibility` | 本地 CosyVoice 模型被误加载 | 查 `.env` 是否 `VOR_TTS_PROVIDER=siliconflow` | 强制使用云端 provider |
| `ModuleNotFoundError: numpy` | 未 `pip install -e .` | `pip list | grep numpy` | 跑安装命令 |
| WebSocket 连上后无响应 | Ollama 未起 | `curl localhost:11434/api/tags` | `systemctl start ollama` |

---

## 九、变更日志

| 版本 | 日期 | 主要变更 |
|------|------|---------|
| v0.x | ~2024-12 | Salesforce VoiceAgentRAG 开源基座 |
| V1.0 | 2025-07 | 业务化改造：SessionContext / IntentRouter / TransferPolicy / ReactAgent / 7 业务工具 / AutoQA |
| V1.5 | 2026-04 | V1 缺陷修复（11 项：并发 Future / Queue 桥接 / 括号平衡 JSON / 括号截断 scratchpad / VIP 规则 / per-session memory ...）|
| v1.0.0 | 2026-04-20 | ASR+TTS 迁移 SiliconFlow 云端；HTTP chunked 流式 TTS；TTFB 14-20s → 800ms |
| **v1.0.1** | **2026-04-20** | 删除 TTS 叠音；IntentRouter 加 "报价/套餐" 关键词；ReactAgent Final Answer 最小长度兜底；文档合并 |

---

*合并自：项目书_语音智能客服系统实施方案.md / V1.0_智能客服系统架构评审与迭代优化白皮书.md / V2.0_本地语音部署与API集成方案.md / 4.20任务_本地全链路部署.md / 启动指南_conda统一环境.md / 环境说明.md*
