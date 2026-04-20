# VoiceAgentRAG 运行说明（Runtime Guide）

## 1. 环境要求

| 项目       | 最低要求              | 推荐                         |
|-----------|---------------------|------------------------------|
| Python    | 3.10+               | 3.11 / 3.12                 |
| OS        | Linux / macOS       | Ubuntu 22.04 + CUDA          |
| GPU       | 可选（无 GPU 降级文本模式） | NVIDIA A10+ (≥12GB VRAM)   |
| 内存       | 8 GB                | 16 GB+                      |
| 磁盘       | 5 GB (模型缓存)       | 20 GB+                      |

---

## 2. 安装

```bash
# 克隆仓库
git clone <repo_url>
cd VoiceAgentRAG

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 安装核心依赖
pip install -e .

# 开发模式（含测试框架）
pip install -e ".[dev]"

# 安装语音模块（需要 GPU 环境）
pip install funasr modelscope torch torchaudio
# CosyVoice 需要单独克隆并安装
# git clone https://github.com/FunAudioLLM/CosyVoice2.git

# Web 服务
pip install fastapi uvicorn websockets

# LLM 供应商（按需安装）
pip install -e ".[openai]"       # OpenAI
pip install -e ".[anthropic]"    # Claude
pip install -e ".[ollama]"       # 本地 Ollama
pip install -e ".[gemini]"       # Google Gemini
```

---

## 3. 配置 (.env)

```bash
cp .env.example .env
# 编辑 .env 填入 API Key 等配置
```

所有配置项均使用 `VOR_` 前缀，通过 Pydantic Settings 读取。核心配置：

| 变量                         | 说明                        | 默认值                  |
|-----------------------------|----------------------------|-----------------------|
| `VOR_LLM_PROVIDER`          | LLM 供应商                  | openai                |
| `VOR_LLM_API_KEY`           | LLM API Key               | (必填)                 |
| `VOR_LLM_MODEL`             | 模型名                      | gpt-4o-mini           |
| `VOR_STT_PROVIDER`          | ASR 供应商                  | sensevoice            |
| `VOR_SENSEVOICE_DEVICE`     | SenseVoice 运行设备          | cuda:0                |
| `VOR_TTS_PROVIDER`          | TTS 供应商                  | cosyvoice             |
| `VOR_COSYVOICE_DEVICE`      | CosyVoice 运行设备           | cuda:0                |
| `VOR_FAISS_INDEX_PATH`      | FAISS 索引路径               | data/faiss_index      |
| `VOR_SESSION_LOG_DIR`       | 会话日志目录                  | logs/sessions         |

**无 GPU 时**：将 `VOR_STT_PROVIDER` 和 `VOR_TTS_PROVIDER` 留空或删除，系统自动降级为纯文本模式。

---

## 4. 程序入口

### 4.1 Web 实时交互服务（推荐）

```bash
# 启动 Web 服务（默认端口 8000）
uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload

# 浏览器访问
open http://localhost:8000
```

Web 服务提供两个 WebSocket 端点：
- `/ws/chat` — 标准交互（整段音频/文本 → 回复）
- `/ws/stream` — **流式语音**（逐 chunk 音频 → partial + final 转写 → 回复）

### 4.2 命令行 Demo

```bash
# 文本交互 CLI（无需语音模块）
python examples/cli_demo.py

# 语音交互 Demo（需要麦克风 + GPU）
python examples/voice_demo.py

# 客服场景演示（完整 pipeline）
python examples/customer_service_demo.py
```

### 4.3 知识库导入

```bash
# 导入文档到 FAISS 知识库
python examples/ingest_documents.py --input-dir ./knowledge_base --index-path data/faiss_index
```

### 4.4 性能基准测试

```bash
python examples/benchmark.py
```

---

## 5. 运行测试

```bash
# 运行全部测试（148 个用例）
pytest

# 运行单个模块
pytest tests/test_streaming_stt.py -v

# 查看每个模块耗时
pytest --tb=short

# 生成覆盖率报告
pytest --cov=voice_optimized_rag --cov-report=html
```

---

## 6. 项目目录结构

```
VoiceAgentRAG/
├── voice_optimized_rag/        # 核心代码包
│   ├── config.py               # Pydantic Settings 全局配置
│   ├── core/                   # 核心框架
│   │   ├── conversation_stream.py  # Event Bus (9 种事件类型)
│   │   ├── semantic_cache.py       # 语义缓存 (embedding 相似度)
│   │   ├── fast_talker.py          # 前台响应代理
│   │   └── slow_thinker.py         # 后台预取代理
│   ├── agent/                  # ReAct Agent 系统
│   │   ├── react_agent.py          # ReAct 推理循环
│   │   ├── permission_guard.py     # 四级权限拦截
│   │   └── tools/                  # 7 个业务工具
│   ├── dialogue/               # 对话管理
│   │   ├── session.py              # 会话状态上下文
│   │   ├── intent_router.py        # 三路意图分类
│   │   ├── emotion_detector.py     # 情绪追踪
│   │   ├── memory_manager.py       # 多轮记忆 (短期+压缩)
│   │   └── transfer_policy.py      # 转人工策略
│   ├── retrieval/              # 知识检索
│   │   ├── vector_store.py         # FAISS 向量数据库
│   │   └── kb_manager.py           # 知识库热更新
│   ├── voice/                  # 语音模块
│   │   ├── stt.py                  # STT 抽象层
│   │   ├── tts.py                  # TTS 抽象层
│   │   ├── sensevoice_stt.py       # SenseVoice + 流式封装
│   │   └── cosyvoice.py            # CosyVoice TTS
│   ├── llm/                    # LLM 抽象层
│   │   └── base.py                 # 多供应商适配
│   └── utils/                  # 工具
│       ├── auto_qa.py              # 自动质检
│       ├── session_logger.py       # JSONL 日志
│       └── logging.py              # 日志工具
├── web/                        # Web 前端 + FastAPI 后端
│   ├── server.py                   # WebSocket 服务端
│   └── static/index.html           # 暗色主题前端 UI
├── tests/                      # 148 个测试用例
├── examples/                   # 示例脚本
├── knowledge_base/             # 知识库文档（待导入）
├── data/                       # FAISS 索引 (运行时生成)
├── logs/                       # 会话日志 (运行时生成)
├── .env.example                # 配置模板
└── pyproject.toml              # 项目元数据 + 依赖
```

---

## 7. 运行模式

### 纯文本模式（开发/测试）

不安装语音模块即可运行。LLM API 正常，语音降级为前端文本输入 + Web Speech API 朗读。

```bash
# .env 中不设置 VOR_STT_PROVIDER / VOR_TTS_PROVIDER
uvicorn web.server:app --port 8000
```

### 完整语音模式（生产）

需要 GPU + SenseVoice + CosyVoice。

```bash
# .env 中设置完整语音配置
VOR_STT_PROVIDER=sensevoice
VOR_SENSEVOICE_DEVICE=cuda:0
VOR_TTS_PROVIDER=cosyvoice
VOR_COSYVOICE_DEVICE=cuda:0

uvicorn web.server:app --host 0.0.0.0 --port 8000 --workers 1
```

> **注意**：语音模型占用 GPU 显存，建议 `--workers 1` 避免多进程加载冲突。

### 流式语音模式

前端通过 `/ws/stream` 端点逐 chunk 发送音频（100ms PCM, 16kHz, mono），服务端实时返回 partial/final 转写 + Agent 回复。

```javascript
// 前端示例（简化）
const ws = new WebSocket('ws://localhost:8000/ws/stream');
mediaRecorder.ondataavailable = (e) => {
    ws.send(e.data);  // 发送二进制 PCM chunk
};
ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'partial') updateTranscript(msg.text);
    if (msg.type === 'final') showFinal(msg.text);
    if (msg.type === 'reply') showReply(msg.text);
};
```

---

## 8. 流式 STT 技术说明

SenseVoice 是**离线模型**（需完整音频段），本系统通过以下策略实现"伪流式"：

```
实时音频 → [100ms chunk] → [Energy VAD] → 语音? → 缓冲
                                           ↓
                           连续说话 1s → Partial 推理 → 中间结果
                                           ↓
                           静音 700ms → Endpoint → Final 推理 → 完整结果
```

**关键参数**（均可配置）：

| 参数                    | 默认值    | 说明                          |
|------------------------|----------|-------------------------------|
| `chunk_duration_ms`    | 100 ms   | 每个输入 chunk 时长             |
| `energy_threshold`     | 0.005    | VAD 能量阈值 (RMS)             |
| `endpoint_silence_ms`  | 700 ms   | 端点检测静音时长                |
| `partial_interval_ms`  | 1000 ms  | Partial 推理间隔               |
| `min_speech_ms`        | 300 ms   | 最小有效语音段                  |
| `max_speech_ms`        | 15000 ms | 最大语音段（强制切割）           |

**延迟分析**：
- Chunk 缓冲延迟: 100ms
- 端点确认延迟: 700ms
- SenseVoice 推理延迟: ~100-300ms (GPU) / ~1-3s (CPU)
- **总端到端延迟**: ~900ms-1.1s (GPU) — 满足实时对话需求 (< 2s)

---

## 9. 常见问题

**Q: 没有 GPU 能跑吗？**
A: 可以。核心 Agent pipeline 只需 LLM API（OpenAI/Claude 等），语音模块自动跳过。Web 界面降级为文本交互。

**Q: `ImportError: funasr` 怎么办？**
A: 语音模块的依赖：`pip install funasr modelscope torch torchaudio`。如果不需要语音功能，不安装即可。

**Q: 测试为什么不需要 GPU？**
A: 所有测试使用 Mock 对象替代真实模型，保证无 GPU 环境也能验证逻辑正确性。

**Q: 如何切换 LLM 供应商？**
A: 修改 `.env` 中的 `VOR_LLM_PROVIDER` (openai/anthropic/ollama/gemini) 和对应的 API Key。

**Q: WebSocket 断开怎么办？**
A: 前端内置自动重连机制（3s 间隔），会话状态是 per-connection 的，重连后开始新会话。

---

## 10. 快速验证

```bash
# 最小化验证（无 GPU, 无 API Key）
pip install -e ".[dev]"
pytest                              # 148 passed

# 有 LLM API Key 时验证 Web 服务
echo "VOR_LLM_API_KEY=sk-xxx" > .env
pip install fastapi uvicorn
uvicorn web.server:app --port 8000  # 访问 http://localhost:8000
```
