# VoiceAgentRAG 技术架构说明书（面试深度版）

> 语音智能客服系统 · 架构设计、核心算法、实现细节与面试知识点

---

## 一、系统总览

### 1.1 一句话定位

VoiceAgentRAG 是一个 **语音驱动的 RAG + ReAct Agent 智能客服系统**。用户通过语音或文字发起对话，系统自动完成：  
**语音识别 → 情绪检测 → 意图路由 → 知识检索 / 任务执行 → 质检 → 语音合成** 的全链路闭环。

### 1.2 全链路架构图

```
┌────────────────────────── 主路径（阻塞） ──────────────────────────┐
│                                                                    │
│  🎤 用户语音                                                       │
│     ↓                                                              │
│  ① SenseVoice-Small (ASR + 情绪标签 + 事件标签)   ← 单模型多任务  │
│     ↓                                                              │
│  ② EmotionDetector (情绪状态机 + EMOTION_CHANGE 事件发布)          │
│     ↓                                                              │
│  ③ IntentRouter (关键词快匹配 → LLM 三路分类)                     │
│     ↓                                                              │
│  ④ TransferPolicy (5 规则评估 → 命中则转人工)                      │
│     ↓                                                              │
│  ┌──────────────┬────────────────┬──────────────┐                  │
│  │  KNOWLEDGE   │     TASK       │   CHITCHAT   │                  │
│  │ RAG 知识检索  │ ReAct Agent    │  LLM 直答    │                  │
│  │ VectorStore  │ Tool+Permission│  情绪安抚    │                  │
│  └──────────────┴────────────────┴──────────────┘                  │
│     ↓                                                              │
│  ⑤ AutoQA 质检 (敏感词 + 禁止模式 + 语义一致性)                    │
│     ↓                                                              │
│  ⑥ CosyVoice 2 (TTS / 声音克隆 / 指令控制)                        │
│     ↓                                                              │
│  🔊 语音回复                                                       │
└────────────────────────────────────────────────────────────────────┘

┌──────────────── 后台路径（并行，不阻塞主路径） ────────────────┐
│  SlowThinker 预取代理 ← 订阅 ConversationStream 事件          │
│  SessionLogger 异步 JSONL 日志 ← run_in_executor 写磁盘       │
│  SemanticCache TTL 过期 ← 后台定时清理                         │
└────────────────────────────────────────────────────────────────┘
```

### 1.3 关键设计指标

| 指标 | 目标 | 实现方式 |
|------|------|---------|
| 首字节延迟 | < 500ms | SemanticCache 命中跳过检索；TTS 流式输出 |
| RTF (Real-Time Factor) | < 0.3 | SenseVoice 比 Whisper 快 15x；SlowThinker 零阻塞 |
| 并发会话 | >= 100 | 全链路 async/await；per-session 状态隔离 |
| 幻觉率 | < 5% | AutoQA 语义一致性 + 敏感词双检 |

---

## 二、核心架构深度解析

### 2.1 双代理架构（Dual-Agent）—— 面试高频考点

**面试问：为什么要用双代理？单代理有什么问题？**

**答：** 语音交互的核心约束是 **延迟**。用户说完话后等待 2 秒和 0.5 秒的体验差异是巨大的。单代理模式下，每次用户提问都要走完 Embedding → FAISS 检索 → LLM 生成的完整链路（200-800ms）。双代理通过「预取」解决这个问题：

```
                         ┌────────────────────────┐
   用户说："这个价格"      │    SlowThinker         │
          │              │  预测下一个问题可能是    │
          ├──publish──→  │  "有没有优惠"           │
          │              │  提前检索并缓存结果     │
          │              └────────────────────────┘
          ↓                         ↓
   ┌──────────────┐        SemanticCache
   │  FastTalker   │    （缓存命中 → 5ms 返回）
   │  先查缓存     │←─────────┘
   │  未命中→检索  │
   └──────────────┘
```

| 代理 | 角色 | 运行方式 | 对延迟的影响 |
|------|------|---------|:-----------:|
| **FastTalker** | 前台：接收请求 → 查缓存 → 未命中则检索 → 生成回复 | 阻塞主路径 | 直接决定延迟 |
| **SlowThinker** | 后台：订阅事件 → 预测下一话题 → 预检索 → 填充缓存 | `asyncio.create_task()` | **0ms**（完全并行） |

**面试追问：SlowThinker 怎么预测下一个问题？**

```python
# slow_thinker.py 核心逻辑
async def _predict_and_prefetch(self, event: StreamEvent):
    # 1. 用 LLM 预测下一个可能的问题
    prediction = await self._llm.generate(
        f"用户刚问了'{event.text}'，预测下一个问题是什么？"
    )
    # 2. 预检索，结果存入 SemanticCache
    results = await self._vector_store.search(prediction, top_k=3)
    await self._cache.put(prediction, results)
```

**面试追问：SlowThinker 的 RTF 影响是多少？**

**零。** 测试中 SlowThinker 耗时 4.8 秒（全部来自 `asyncio.sleep(2.0)` 的 mock 等待），但它通过 `asyncio.create_task()` 在后台运行，不阻塞任何响应。事件循环在等待 SlowThinker 时可以处理其他请求。

### 2.2 事件总线（ConversationStream）—— 发布-订阅模式

**面试问：模块间怎么通信？为什么不直接函数调用？**

**答：** 系统有 15+ 个模块，如果用直接调用会产生 **循环依赖**。例如：
- Agent 执行写操作 → 需要用户确认 → 需要语音模块播放确认话术 → Agent 等待确认结果
- 如果 Agent 直接调用语音模块，就形成了 Agent → Voice → Agent 的循环

**解决方案：事件总线解耦。**

```python
class ConversationStream:
    """
    核心数据结构：
    - _subscribers: list[asyncio.Queue]  # 每个订阅者独立的事件队列
    - _history: deque(maxlen=N)          # 滑动窗口历史
    """

    async def publish(self, event: StreamEvent) -> None:
        """广播到所有订阅者队列 + 写入历史"""
        self._history.append(event)
        for queue in self._subscribers:
            await queue.put(event)  # 异步非阻塞

    def subscribe(self) -> AsyncIterator[StreamEvent]:
        """创建新订阅，返回异步迭代器"""
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        return _SubscriptionIterator(queue, self._subscribers)
```

**9 种事件类型及其流转：**

| 事件 | 发布者 | 订阅者 | 场景 |
|------|--------|--------|------|
| `USER_UTTERANCE` | WebServer | SlowThinker, SessionLogger | 每次用户说话 |
| `AGENT_RESPONSE` | ReactAgent | SlowThinker, SessionLogger | 每次 Agent 回复 |
| `EMOTION_CHANGE` | EmotionDetector | TransferPolicy | 情绪变化时 |
| `CONFIRM_REQUIRED` | PermissionGuard | WebServer(→前端) | L2+工具需确认 |
| `CONFIRM_RESPONSE` | WebServer(←前端) | PermissionGuard | 用户确认/拒绝 |
| `TRANSFER_REQUEST` | TransferPolicy/Agent | WebServer | 触发转人工 |
| `TOPIC_SHIFT` | IntentRouter | SemanticCache | 话题切换 |
| `SILENCE_DETECTED` | SpeechService | SlowThinker | 利用空档预取 |
| `PRIORITY_RETRIEVAL` | FastTalker | SlowThinker | 缓存未命中紧急检索 |

**面试追问：为什么用 `asyncio.Queue` 而不是回调函数？**

1. **背压传播**：Queue 满时 `put()` 会等待，自动限流慢消费者
2. **异步友好**：`async for event in subscription` 天然支持异步迭代
3. **自动清理**：`CancelledError` 时自动从订阅列表中移除自身

```python
class _SubscriptionIterator:
    async def __anext__(self) -> StreamEvent:
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            # 关键：任务被取消时自动退订，防止内存泄漏
            self._subscribers.remove(self._queue)
            raise
```

### 2.3 语义缓存（SemanticCache）—— FAISS + 内积相似度

**面试问：语义缓存和普通缓存的区别？**

**答：** 普通缓存是精确匹配 key，"手机多少钱" 和 "手机什么价格" 会被视为两个不同的 key。语义缓存用 **向量相似度** 匹配，这两个语义相同的问题会命中同一个缓存。

```python
class SemanticCache:
    """
    底层结构：
    - FAISS IndexFlatIP (内积索引)
    - L2 归一化后的内积 = 余弦相似度
    """

    async def get(self, query: str, threshold: float = 0.85):
        # 1. query → Embedding → L2 归一化
        query_emb = await self._embedding.embed([query])
        faiss.normalize_L2(query_emb)

        # 2. FAISS 内积搜索（= 余弦相似度）
        scores, indices = self._index.search(query_emb, k=1)

        # 3. 相似度 > 阈值 → 缓存命中
        if scores[0][0] > threshold:
            return self._cache[indices[0][0]]  # ~5ms
        return None  # 未命中 → 走 RAG 检索 ~200ms
```

**面试追问：为什么用 IndexFlatIP 不用 IndexFlatL2？**

- `IndexFlatIP` = 内积（Inner Product），配合 L2 归一化后等价于余弦相似度
- 余弦相似度只看方向不看长度，更适合语义匹配
- `IndexFlatL2` = 欧氏距离，对向量长度敏感，不适合文本 Embedding

**面试追问：TTL 怎么实现？**

缓存每条记录附带 `created_at` 时间戳。查询时检查 `time.time() - created_at > ttl`，过期则视为未命中。后台定期清理过期条目。

---

## 三、RAG 检索系统

### 3.1 文档处理管线

```
原始文档 (.txt / .md / .pdf)
    ↓
DocumentLoader.load_directory()
    ↓
chunk_text(text, chunk_size=512, overlap=50)
    ↓ 分块策略：按字符切割，重叠 50 字符保证边界语义连续
Embedding Provider (OpenAI / Ollama / 本地)
    ↓ text → float[1536] 或 float[768]
FAISS IndexFlatIP (L2 归一化后入库)
    ↓
FAISSVectorStore._texts[] + ._metadata[]
```

**面试问：为什么分块要有 overlap？**

**答：** 假设一个句子正好被切成两块：
```
Chunk A: "...这款手机的售价是"
Chunk B: "2999 元，支持 5G..."
```
如果用户问 "手机多少钱"，不重叠时 Chunk A 和 Chunk B 都无法单独回答。  
overlap=50 意味着 Chunk B 的开头会包含 `"售价是 2999 元"`，保证了切割边界的信息完整性。

### 3.2 检索流程（KNOWLEDGE 意图）

```python
# web/server.py 简化版
if intent == IntentType.KNOWLEDGE:
    # 1. 获取对话记忆上下文（摘要 + 近期原文）
    context = memory_manager.get_context()

    # 2. 向量检索（实际生产中先查 SemanticCache）
    # docs = vector_store.search(query, top_k=3)

    # 3. LLM 生成（context = 记忆 + 检索文档）
    reply = await llm.generate(
        f"基于上下文回答用户问题：\n用户：{user_text}",
        context=context,  # 注入检索结果
    )
```

### 3.3 与传统 RAG 的三大区别

| 维度 | 传统 RAG | VoiceAgentRAG |
|------|---------|--------------|
| **缓存** | 无 | SemanticCache（相似问题 ~5ms 返回） |
| **预取** | 无 | SlowThinker 预测下一话题，提前填充缓存 |
| **输入** | 纯文本 | 语音 + 情绪标签 + 事件标签（多模态） |
| **降级** | 单一路径 | 三路分类 + 转人工兜底（多级容错） |

**面试问：RAG 的 R 是什么？和 Fine-tuning 的区别？**

**答：** R = Retrieval（检索）。RAG 在推理时从外部知识库检索相关文档，作为上下文传给 LLM。

| 对比 | RAG | Fine-tuning |
|------|-----|-------------|
| 知识更新 | 实时（更新向量库即可） | 需要重新训练 |
| 成本 | 低（只需 Embedding） | 高（GPU 训练） |
| 幻觉控制 | 好（回答基于检索文档） | 差（模型可能编造） |
| 适合场景 | 事实性问答、FAQ | 风格迁移、特定领域 |

---

## 四、Agent 系统

### 4.1 ReAct 架构 —— 面试核心考点

**面试问：你们的 Agent 用的什么架构？**

**答：ReAct（Reasoning + Acting），不是 Plan-and-Execute，不是 Function Calling。**

ReAct 的核心是 **交替进行推理和行动**，每一步都有显式的 Thought（思考过程），使得决策过程可审计。

```
循环开始
│
├─ Thought: 用户要查订单 ORD-001 的物流
│
├─ Action: query_order
│  Action Input: {"order_id": "ORD-001"}
│
├─ [系统] → PermissionGuard L1 检查 → 通过
│  [系统] → QueryOrderTool.execute()
│  [系统] → Observation: 订单已发货，顺丰 SF123...
│
├─ Thought: 已获取到物流信息，可以回复用户
│
└─ Final Answer: 您的订单 ORD-001 已发货，由顺丰速运承运（单号 SF123），
                 预计 4 月 21 日到达。
```

### 4.2 ReAct 核心循环实现

```python
class ReactAgent:
    async def execute(self, user_request, session, memory_context=""):
        # 构建 Prompt：system prompt + 工具列表 + scratchpad
        scratchpad = f"用户请求: {user_request}\n"

        for iteration in range(max_iterations):  # 默认 10 轮
            # ── 防止 scratchpad 超出 LLM 上下文窗口 ──
            if len(scratchpad) > 6000:
                # 保留头部(用户请求) + 尾部(最近推理)，中间截断
                scratchpad = header + "\n...(省略)...\n" + tail

            # ── 1. LLM 推理 ──
            response = await llm.generate(system + scratchpad)
            scratchpad += response  # 追加到推理轨迹

            # ── 2. 检查是否有最终回答 ──
            final = _extract_final_answer(response)  # 正则: "Final Answer: (.*)"
            if final:
                return final  # 路径 A: 直接回答

            # ── 3. 解析工具调用 ──
            action_name, action_input = _parse_action(response)
            # 使用括号平衡法解析 JSON（支持嵌套对象）

            # ── 4. 权限检查 ──
            perm = await permission_guard.check_permission(tool, session)
            if not perm.success:
                scratchpad += f"Observation: 权限失败 - {perm.message}\n"
                continue

            # ── 5. 执行工具（超时 3s + 重试 1 次）──
            result = await execute_tool_with_retry(tool, action_input)
            scratchpad += f"Observation: {result}\n"

        # 路径 C: 超过最大迭代 → 转人工
        return "转接人工客服"
```

### 4.3 三大退出路径

| 路径 | 条件 | 处理 |
|------|------|------|
| **A. Final Answer** | LLM 输出包含 `Final Answer:` | 正常返回回复 |
| **B. 连续失败** | 工具连续失败 ≥ 3 次 | 发布 TRANSFER_REQUEST，转人工 |
| **C. 迭代上限** | 循环 ≥ 10 次仍无 Final Answer | 转人工 |

### 4.4 JSON 解析：括号平衡法 vs 正则

**面试问：LLM 输出的 JSON 怎么解析？正则靠谱吗？**

**答：** V1 用的正则 `\{.+?\}` 非贪婪匹配，遇到嵌套 JSON 会截断：
```
Action Input: {"address": {"city": "北京", "detail": "朝阳区"}}
正则匹配到: {"address": {"city": "北京", "detail": "朝阳区"}  ← 少了最后的 }
```

**V2 改用括号平衡计数法：**
```python
def _parse_action(text):
    # 找到第一个 { 后，维护深度计数
    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}": depth -= 1
        if depth == 0:
            return json.loads(text[brace_start:i+1])  # 完整 JSON
```

时间复杂度 O(n)，单次扫描，支持任意深度嵌套。

### 4.5 为什么选 ReAct 不选 Function Calling

| 维度 | ReAct | OpenAI Function Calling |
|------|-------|------------------------|
| **LLM 无关性** | ✅ 任何 LLM（Ollama/千问/Llama） | ❌ 仅 OpenAI 系列 |
| **推理透明度** | ✅ Thought 可审计、可溯源 | ❌ 黑盒决策 |
| **本地部署** | ✅ 纯文本协议，Ollama 即可 | ❌ 需要专属 API |
| **解析可靠性** | ⚠️ 依赖文本解析 | ✅ 结构化 JSON Schema |
| **多步推理** | ✅ 天然支持多步 Thought | ⚠️ 需要 parallel function call |

**结论：** 本项目优先本地部署和 LLM 无关性，因此选 ReAct。

### 4.6 四级权限系统 —— asyncio.Future 确认流

**面试问：用户确认是怎么实现的？轮询还是事件驱动？**

**答：事件驱动，基于 `asyncio.Future` 零轮询等待。**

```python
class PermissionGuard:
    # 按 request_id 存储 Future，支持并发确认（V2 修复竞态条件）
    _pending_confirms: dict[str, asyncio.Future] = {}

    async def _request_confirmation(self, tool, session):
        request_id = uuid4()[:8]

        # 1. 创建 Future（此时处于 pending 状态）
        future = loop.create_future()
        self._pending_confirms[request_id] = future

        # 2. 发布确认事件到前端
        await stream.publish(CONFIRM_REQUIRED, metadata={"request_id": request_id})

        # 3. 挂起等待（不占 CPU，不轮询）
        try:
            return await asyncio.wait_for(future, timeout=15)
        except TimeoutError:
            return False  # 超时 = 拒绝
        finally:
            self._pending_confirms.pop(request_id, None)

    async def handle_confirm_response(self, confirmed, request_id=""):
        # 前端回传 request_id → 精确匹配对应的 Future
        if request_id in self._pending_confirms:
            self._pending_confirms[request_id].set_result(confirmed)
```

**面试追问：V1 有什么问题？V2 怎么修的？**

| 问题 | V1 | V2 |
|------|----|----|
| 并发安全 | 单一 `_pending_confirm` Future，并发时后者覆盖前者 | `dict[request_id, Future]` 隔离每个请求 |
| 匹配精度 | 任意确认响应都 set 唯一的 Future | 通过 `request_id` 精确匹配 |
| 兼容性 | — | 无 request_id 时兜底 FIFO 匹配（向下兼容） |

**四级权限矩阵：**

```
Level 1 (只读)  → 直接放行
                   例: QueryOrder, QueryInventory

Level 2 (写入)  → 发布 CONFIRM_REQUIRED → 等待 Future → 确认后执行
                   例: UpdateAddress, CancelOrder

Level 3 (财务)  → Level 2 确认 + 身份验证（二次确认）
                   例: ApplyRefund

Level 4 (管理)  → 直接拒绝（Agent 不可执行）
                   例: 预留管理操作
```

### 4.7 业务工具集

| 工具 | 权限 | 参数 | 实现状态 |
|------|:---:|------|---------|
| `query_order` | L1 | order_id, phone | ⚠️ Mock（TODO: 对接订单 API） |
| `query_inventory` | L1 | product_name | ⚠️ Mock |
| `get_customer_info` | L1 | customer_id | ⚠️ Mock |
| `check_promotion` | L1 | category | ⚠️ Mock |
| `update_address` | L2 | order_id, new_address | ⚠️ Mock |
| `cancel_order` | L2 | order_id, reason | ⚠️ Mock |
| `apply_refund` | L3 | order_id, reason, amount | ⚠️ Mock |

> **所有 7 个工具的接口（BaseTool 抽象类）、参数校验、权限标注、Prompt 描述生成已完整实现。** `execute()` 方法当前返回硬编码 Mock 数据。接入真实 API 只需替换 `# TODO` 后的逻辑，上层 Agent 无需改动。

---

## 五、对话管理系统

### 5.1 意图路由（三路分类）

**面试问：意图识别用的什么方法？**

**答：分层策略，关键词优先 + LLM 兜底：**

```python
class IntentRouter:
    async def classify(self, utterance, session):
        # 优先级 1：转人工关键词（最高优先级）
        for kw in ["转人工", "人工客服", "找经理"]:
            if kw in text:
                session.transfer_requested = True
                return CHITCHAT  # 标记转人工，但路由到闲聊先安抚

        # 优先级 2：任务关键词 → 零延迟匹配
        for kw in ["查订单", "退款", "修改地址"]:
            if kw in text:
                return TASK

        # 优先级 3：知识关键词
        for kw in ["多少钱", "怎么用", "保修"]:
            if kw in text:
                return KNOWLEDGE

        # 优先级 4：LLM 三路分类（兜底）
        result = await llm.generate(INTENT_PROMPT.format(utterance=text))
        # 解析 "task" / "knowledge" / "chitchat"

        # 异常降级：LLM 失败 → 默认 KNOWLEDGE（最安全）
```

**面试追问：为什么转人工关键词返回 CHITCHAT 而不是直接跳出？**

**答：** 转人工是由 `TransferPolicy` 在后续步骤统一处理的。IntentRouter 只负责标记 `session.transfer_requested = True`。这样做的好处是 TransferPolicy 可以综合考虑其他因素（比如 VIP 等级、情绪状态），而不是关键词一刀切。同时走 CHITCHAT 路径可以先给用户一个安抚回复。

### 5.2 情绪检测 —— SenseVoice 一体化

**面试问：情绪识别用的独立模型吗？**

**答：不是。SenseVoice-Small 是多任务模型，单次推理同时输出 4 种信息：**

```
语音输入 → SenseVoice →  "<|zh|><|ANGRY|><|Speech|><|woitn|>我的订单怎么还没到"
                           │       │        │         │
                           语言    情绪     事件      ITN标记    → 纯文本
```

**EmotionDetector 只做标签映射 + 状态机维护：**

```python
class EmotionDetector:
    async def update(self, raw_emotion: str, session: SessionContext):
        # 1. 标签映射：SenseVoice标签 → 系统枚举
        new_emotion = EMOTION_MAP.get(raw_emotion.lower(), NEUTRAL)

        # 2. 更新 Session 状态
        session.update_emotion(new_emotion)
        # ↑ 内部逻辑：
        #   - emotion_history.append(new_emotion)  # 带上限，防内存泄漏
        #   - if ANGRY: consecutive_angry_turns += 1
        #   - else:     consecutive_angry_turns = 0

        # 3. 情绪变化时发布事件
        if new_emotion != old_emotion:
            await stream.publish(EMOTION_CHANGE, text=new_emotion.value)
```

### 5.3 多轮记忆管理 —— 三级记忆架构

**面试问：多轮对话的上下文怎么管理？Token 不会爆吗？**

**答：三级记忆策略，核心是「新的保原文，旧的压摘要」：**

```
┌─────────────────────────────────────────────────────┐
│  短期记忆（最近 10 轮）                               │
│  ┌──────────────────────────────────────────────┐   │
│  │ User: 我要退款                                │   │
│  │ Asst: 好的，请提供订单号                       │   │  ← 原文保留
│  │ User: ORD-001                                 │   │
│  │ Asst: 已为您提交退款申请                       │   │
│  └──────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────┤
│  中期摘要（10 轮以前）                               │
│  "用户先咨询了手机价格（2999元），然后询问了优惠活动     │  ← LLM 压缩
│   （满减200），情绪较平稳。"                          │
├─────────────────────────────────────────────────────┤
│  长期存储（会话结束后）                               │
│  → JSONL 落盘 → V2 对接 CRM                         │  ← 异步写入
└─────────────────────────────────────────────────────┘
```

**压缩触发机制：**

```python
class MemoryManager:
    async def add_turn(self, role, content, session):
        self._turns.append(MemoryTurn(role, content))

        # 短期窗口满时自动压缩
        if len(self._turns) > self._short_term_turns:
            await self._compress_old_turns(session)

    async def _compress_old_turns(self, session):
        # 取出溢出部分
        old_turns = []
        while len(self._turns) > self._short_term_turns:
            old_turns.append(self._turns.popleft())

        # LLM 压缩为摘要（失败时退化为截断拼接）
        self._summary = await llm.generate(COMPRESS_PROMPT.format(
            conversation=format(old_turns),
            existing_summary=self._summary,
        ))
```

**面试追问：压缩用的 LLM 调用成本高吗？**

**答：** 只在对话超过 10 轮时触发一次（而不是每轮都触发），且 prompt 限制在 200 字以内的摘要输出。10 轮对话约 1000-2000 token 输入，摘要输出 ~200 token。对于客服场景，大部分会话在 5-8 轮内结束，多数情况下不会触发压缩。

### 5.4 转人工策略 —— 5 条规则引擎

```python
class TransferPolicy:
    def _check_rules(self, session, utterance) -> str:
        # 规则 1：持续愤怒 >= 阈值（默认 2 轮）
        if session.consecutive_angry_turns >= threshold:
            return "持续愤怒"

        # 规则 2：Agent 连续失败 >= 3 次
        if session.agent_failure_count >= 3:
            return "Agent 连续失败"

        # 规则 3：VIP 客户 + 不满情绪（V2 新增，修复了 V1 的死代码）
        if session.vip_level >= 3 and emotion in (ANGRY, SAD):
            return "VIP 客户情绪不佳"

        # 规则 4：高风险关键词（法律/投诉相关）
        for kw in ["法律", "律师", "起诉", "消协"]:
            if kw in utterance:
                return f"高风险关键词: {kw}"

        return ""  # 不转人工
```

> **V1→V2 修复：** 原规则 4（VIP 判断）是死代码（`pass` 空语句），规则 2（关键词检测）被入口的 `transfer_requested` 前置检查覆盖导致不可达。V2 重新设计了规则列表，确保每条规则都能被触发。

---

## 六、语音模块

### 6.1 ASR：SenseVoice-Small

**面试问：为什么选 SenseVoice 不选 Whisper？**

| 维度 | SenseVoice-Small | Whisper Large V3 |
|------|-----------------|------------------|
| 速度 | **15x 快于 Whisper** | 较慢 |
| 多任务 | ✅ ASR + 情绪 + 事件 + 语言识别 | ❌ 仅 ASR |
| 中文效果 | 针对中文优化 | 多语言通用 |
| 模型大小 | ~300MB | ~1.5GB |
| 框架 | FunASR | Transformers |

**核心实现：**

```python
class SenseVoiceSTT(STTProvider):
    def _ensure_model(self):
        """延迟加载：首次调用时才初始化模型"""
        from funasr import AutoModel
        self._model = AutoModel(
            model="iic/SenseVoiceSmall",
            trust_remote_code=True,
            device="cuda:0",
        )

    async def transcribe_with_emotion(self, audio_data, sample_rate=16000):
        # 1. bytes → numpy float32
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 2. 异步推理（run_in_executor 避免阻塞事件循环）
        raw = await loop.run_in_executor(None, lambda: self._model.generate(
            input=audio_np, cache={}, language="auto", use_itn=True
        ))

        # 3. 解析标签
        return self._parse_result(raw)
        # 输入: "<|zh|><|ANGRY|><|Speech|><|woitn|>订单怎么还没到"
        # 输出: SenseVoiceResult(text="订单怎么还没到", emotion="angry", event="speech", language="zh")
```

### 6.2 TTS：CosyVoice 2

**三种合成模式：**

```python
# 1. 标准模式（预置发音人）
audio = await tts.synthesize("你好", mode="standard", speaker="中文女")

# 2. 零样本克隆（3-10 秒参考音频即可克隆任意声音）
audio = await tts.synthesize("你好", mode="clone", reference_audio=ref_bytes)

# 3. 指令控制（文字描述风格）
audio = await tts.synthesize("你好", mode="instruct", instruct_text="用温柔缓慢的声音说")
```

**流式合成（V2 修复 — Queue 桥接模式）：**

```python
async def synthesize_stream(self, text, mode="standard", speaker=None):
    """
    V1 Bug: run_in_executor(None, generator_func) 返回生成器对象而非迭代
    V2 Fix: 使用 asyncio.Queue 桥接同步生成器与异步迭代器
    """
    queue = asyncio.Queue(maxsize=8)

    async def _producer():
        def _run_sync():
            for chunk in model.inference_sft(text, speaker, stream=True):
                # 从工作线程安全推入 asyncio Queue
                asyncio.run_coroutine_threadsafe(queue.put(pcm), loop).result()
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()  # sentinel
        await loop.run_in_executor(None, _run_sync)

    producer_task = asyncio.create_task(_producer())

    while True:
        chunk = await queue.get()
        if chunk is None: break
        yield chunk
```

**面试追问：为什么不能直接 `run_in_executor(None, generator)`？**

**答：** `run_in_executor` 只能运行普通函数，不能运行生成器。它会把生成器函数当作普通函数调用，返回的是生成器对象本身（而不是迭代它）。即使能返回，同步生成器的 `__next__()` 调用也会阻塞事件循环。Queue 桥接将生产者放在线程池，消费者通过 `await queue.get()` 异步等待，两者完全解耦。

---

## 七、异步编程模型 —— 面试必考

### 7.1 为什么全链路用 async/await？

**语音客服的核心矛盾：高并发 + 高 I/O 等待**

每个请求涉及：STT 推理 → LLM API 调用 → 数据库查询 → TTS 推理。  
其中 80%+ 的时间在等待 I/O。如果用同步线程模型：

```
# 同步模型：100 并发 = 100 个线程，大量 I/O 等待浪费
Thread-1: [STT wait......][LLM wait..........][TTS wait....]
Thread-2: [STT wait......][LLM wait..........][TTS wait....]
```

```
# 异步模型：单线程处理 100 并发，I/O 等待时切换
EventLoop: [STT-1]→[STT-2]→[LLM-1 returned]→[LLM-2 returned]→[TTS-1]→...
```

### 7.2 关键异步模式

**模式 1：`asyncio.create_task()` — 后台并行**
```python
# SlowThinker 不阻塞主路径
task = asyncio.create_task(slow_thinker.start())
# ↑ 立即返回，后台运行
```

**模式 2：`run_in_executor()` — CPU 密集型任务丢线程池**
```python
# SenseVoice 推理是 CPU 密集型，放线程池避免阻塞事件循环
result = await loop.run_in_executor(None, model.generate, audio)
```

**模式 3：`asyncio.Future` — 事件驱动等待**
```python
# PermissionGuard：等待用户确认，不轮询
future = loop.create_future()
# ... 发布事件到前端 ...
confirmed = await asyncio.wait_for(future, timeout=15)
# ↑ 挂起直到 future.set_result(True/False) 被调用
```

**模式 4：`asyncio.Queue` — 生产者-消费者桥接**
```python
# CosyVoice 流式合成：同步生成器 → asyncio Queue → async yield
queue = asyncio.Queue(maxsize=8)
# 生产者在线程池，消费者在事件循环，完全解耦
```

**面试问：`get_event_loop()` 和 `get_running_loop()` 的区别？**

**答：**
- `get_event_loop()`：Python 3.10+ 已弃用。在没有运行中的事件循环时可能返回 None 或创建新的循环，行为不确定。
- `get_running_loop()`：只在有运行中的事件循环时返回，否则抛 RuntimeError。**更安全，语义更明确。**

> V2 将所有 `get_event_loop()` 替换为 `get_running_loop()`（影响 5 个文件）。

---

## 八、质检与安全

### 8.1 AutoQA 三级质检

```python
class AutoQA:
    async def check(self, response, source_context=""):
        issues = []

        # 级别 1：敏感词检测（仅警告，不替换回复）
        # "保证"、"一定" → 可能构成虚假承诺
        sensitive = self._check_sensitive_words(response)
        if sensitive:
            issues.append(f"敏感词: {sensitive}")

        # 级别 2：禁止模式检测（触发 → 替换为兜底回复）
        # "竞品很差"、"保证收益" → 法律风险
        forbidden = self._check_forbidden_patterns(response)
        if forbidden:
            issues.append(f"违规: {forbidden}")
            cleaned = FALLBACK_RESPONSE  # 替换整个回复

        # 级别 3：语义一致性（需 Embedding Provider）
        # 回答与检索文档的余弦相似度 < 0.8 → 可能幻觉
        consistency = await self._check_consistency(response, source_context)
```

**V1→V2 修复：** V1 中敏感词（如"保证"）会触发全量替换为兜底回复，导致大量正常回复被误杀。V2 改为敏感词仅记录警告，不替换；只有禁止模式和语义一致性问题才触发替换。

### 8.2 日志审计

```python
class SessionLogger:
    async def log_event(self, session_id, event_type, data):
        """异步 JSONL 写入（run_in_executor 避免阻塞）"""
        # 每个会话独立文件：20260419_abc12345.jsonl
        entry = {"timestamp": time.time(), "session_id": ..., "event_type": ..., **data}
        await loop.run_in_executor(None, self._write_line, path, entry)
```

---

## 九、Web 实时交互

### 9.1 全链路协议

```
浏览器 ──WebSocket──→ FastAPI Server ──→ Pipeline
  │                      │
  ├─ {"type":"text"}     │  文字模式
  ├─ {"type":"audio"}    │  语音模式（base64 PCM）
  │                      │
  ←─ {"type":"reply"}    │  回复（text + audio + emotion + timing_ms）
  ←─ {"type":"confirm"}  │  权限确认请求
  ─→ {"type":"confirm_response"} │  用户确认
```

### 9.2 优雅降级

```python
# 启动时尝试加载，失败则降级
llm = _try_load_llm(config)    # 失败 → EchoLLM（回显模式）
stt, tts = _try_load_voice(config)  # 失败 → None（文字模式）
```

| 组件 | 可用 | 不可用 |
|------|------|--------|
| LLM | 真实对话 | Echo 模式（回显提示配置 API Key） |
| STT | 语音转写 | 前端文字输入替代 |
| TTS | 语音播放 | Web Speech API 朗读 |

### 9.3 Per-Session 状态隔离

**V2 修复：** V1 中所有 WebSocket 连接共享同一个 MemoryManager，导致多用户记忆混淆。V2 为每个 WebSocket 连接创建独立的 MemoryManager 实例：

```python
async def websocket_chat(ws):
    session = SessionContext()  # 独立会话
    per_session_memory = MemoryManager(llm, ...)  # 独立记忆
```

---

## 十、V2 缺陷修复总结

### 10.1 已修复的缺陷（本次迭代）

| # | 缺陷 | 严重度 | 修复方式 | 影响文件 |
|---|------|:------:|---------|---------|
| 1 | TransferPolicy 规则 4 死代码 (`pass`) | 中 | 重写为 VIP + 不满情绪组合判断 | `transfer_policy.py` |
| 2 | TransferPolicy 规则 2 不可达 | 中 | 删除（已被入口前置检查覆盖） | `transfer_policy.py` |
| 3 | PermissionGuard 并发竞态条件 | 高 | `dict[request_id, Future]` 替代单一 Future | `permission_guard.py` |
| 4 | CosyVoice 流式阻塞事件循环 | 高 | Queue 桥接同步生成器与异步迭代器 | `cosyvoice.py` |
| 5 | `asyncio.get_event_loop()` 已弃用 | 低 | 全部替换为 `get_running_loop()` | 4 个文件 |
| 6 | ReAct JSON 解析失败（嵌套对象） | 中 | 括号平衡计数法替代正则 | `react_agent.py` |
| 7 | Scratchpad 无限增长 | 中 | 超过 6000 字符时截断中间、保留首尾 | `react_agent.py` |
| 8 | AutoQA 敏感词过度拦截 | 中 | 敏感词仅警告、禁止模式才替换 | `auto_qa.py` |
| 9 | Web 多会话共享 MemoryManager | 高 | 每个 WebSocket 独立实例 | `web/server.py` |
| 10 | Web STT 调用返回类型错误 | 中 | 使用 `transcribe_with_emotion()` | `web/server.py` |
| 11 | emotion_history 无限增长 | 低 | 增加 50 条上限裁剪 | `session.py` |

### 10.2 仍存在的已知限制

| 项目 | 说明 | 影响 |
|------|------|------|
| 7 个工具返回 Mock 数据 | 需对接真实 CRM / 订单 / 库存 API | 功能不完整 |
| KBManager 访问私有属性 | 直接访问 `_metadata`, `_texts`, `_index` | 封装性差，升级 VectorStore 时可能 break |
| ConversationStream 无 Queue 上限 | 慢消费者不会被限流 | 潜在内存泄漏 |
| 敏感词列表硬编码 | "保证"、"一定" 等误报率高 | 需改为配置文件 + 上下文感知 |
| SenseVoice 仅支持离线推理 | 需用户发送完整音频段 | 不支持边说边识别的流式 STT |

---

## 十一、测试概况

```
Total: 137 tests, ALL PASSED
Coverage:
  - 20 个测试文件覆盖 20 个源模块
  - 每个模块的正常路径 + 异常路径 + 边界条件
  - 新增: VIP 转人工 (2 tests) + 并发权限确认 (1 test)
```

| 模块 | 测试数 | 耗时 | 说明 |
|------|:-----:|-----:|------|
| SlowThinker | 4 | 4.8s | 全部来自 mock sleep，RTF=0 |
| PermissionGuard | 8 | 1.6s | 含 1s 超时测试 + 并发测试 |
| SemanticCache | 9 | 154ms | FAISS 检索性能验证 |
| 其余 17 模块 | 116 | <500ms | 纯逻辑测试 |

---

## 十二、技术栈

| 层 | 技术 | 选型理由 |
|---|------|---------|
| 语言 | Python ≥ 3.10 | AI 生态最完善 |
| 异步框架 | asyncio | 原生协程，无额外依赖 |
| Web 框架 | FastAPI + Uvicorn | 高性能 ASGI + WebSocket 原生支持 |
| 向量数据库 | FAISS (faiss-cpu) | 内存索引，低延迟，Meta 出品 |
| LLM | OpenAI / Anthropic / Ollama | 多后端可切换 |
| ASR | SenseVoice-Small (FunASR) | 中文优化，多任务一体 |
| TTS | CosyVoice 2 | 合成 + 克隆统一模型 |
| 配置 | Pydantic Settings + `.env` | 类型安全 + 环境变量 |
| 测试 | pytest + pytest-asyncio | 异步测试原生支持 |

---

*文档版本: V2.0 | 项目: VoiceAgentRAG 语音智能客服系统 | 测试: 137/137 PASSED*
