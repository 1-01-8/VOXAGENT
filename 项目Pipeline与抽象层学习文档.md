# 项目 Pipeline 与抽象层学习文档

## 1. 现在这套系统是什么

当前仓库已经不是最初那种“知识问答 + ReAct + 语音链路并列存在”的松散组合，而是逐步收敛成一套纯业务导向的客服系统。

如果只用一句话概括当前状态：

> 这是一个面向销售、售后、财务三类业务的客服系统，知识问题走混合检索 RAG，部分高价值操作先走显式任务状态机，其余操作再走原生 Function Calling Agent，非业务请求统一走 out_of_scope 拒答，本地 CLI 通过 Ollama 直接测试，并且可以打印模块决策轨迹和 LLM 调用轨迹。

当前最重要的变化是：

1. 业务执行链已经从“只有 Function Calling Agent”升级为“显式任务状态机 + Function Calling fallback”。
2. 销售、售后、财务三层已经从“领域 prompt 约束”升级成正式 `SkillRegistry`。
3. 销售知识回答不再只是通用回答 prompt，而是补上了结构化价格模板、产品介绍模板、商品目录模板、实体说明模板和短型号查询模板。
4. 原来的 `chitchat` 设计已经从主业务链里移除，改为 `out_of_scope`。
5. 新增了 `knowledge_base/real_world/` 公开业务数据摘要，并修复了入库脚本与去重链路，便于真实 RAG 测试。
6. 本地纯文本业务 CLI 不只用于测试业务能力，也已经成为当前最强的调试入口：可打印模块决策轨迹和 LLM 调用轨迹。
7. 新增了业务型 MCP server，并且 MCP / Web / CLI 的任务入口都已接上显式任务状态机。


## 1.1 可直接写进简历的自然语言版本

### 简历短版

我设计并落地了一套面向销售、售后、财务场景的多 Agent 智能客服系统。系统不是把所有请求直接丢给单一大模型，而是先做意图分流，再按业务域把请求调度到对应 Agent；对于退款、取消订单、修改地址这类高风险操作，优先使用显式状态机和权限确认机制保证流程可控，其余任务再交给原生 Function Calling Agent 执行。知识问答侧则基于混合 RAG 方案，结合向量检索、BM25 倒排、RRF 融合和启发式 rerank，配合语义缓存与真实业务知识库，提升了中文业务问答的召回效果、稳定性和可解释性。

### 简历标准版

在这个项目中，我负责把传统“单 Agent + 通用 prompt”的客服形态，重构为一套具备多重调度能力的业务 Agent 系统。用户请求会先被拆分为知识问答、任务办理和超范围请求，再根据销售、售后、财务三个业务域做第二层路由；这样做的好处是，模型不再承担全部流程控制，而是由路由层、技能注册表和状态机共同决定该由谁回答、该调用什么工具、什么时候需要追问、什么时候需要确认。对退款、取消订单、修改地址等关键流程，我引入了显式任务状态机、槽位同步和权限校验，减少模型自由发挥带来的不确定性；对一般业务办理，则采用原生 Function Calling，并保留 ReAct 作为兼容回退，使系统在不同模型能力下都能稳定运行。

知识侧我重点完善了 RAG 中台，而不是只做简单向量检索。整体链路由 `MemoryRouter` 统一协调，前台由 `FastTalker` 负责低延迟响应，后台由 `SlowThinker` 做预取和缓存预热；检索层采用 dense 向量召回和 sparse 倒排 BM25 双路召回，再通过 RRF 融合和启发式 rerank 提高相关性，同时结合中文查询到英文知识库的别名扩展，解决中文业务问法和英文资料之间的语义错位问题。除此之外，我还补充了真实公开业务数据的入库、去重和调试链路，让系统不仅能回答内部产品知识，也能回答 Zendesk、Pipedrive、Stripe Billing 等外部业务平台的结构化问题。

### 面试展开版

如果用更适合面试或项目介绍的方式来概括，这个项目的核心价值不在于“接了多少模型”，而在于我把 Agent 从单点问答能力，做成了一套可调度、可约束、可观测的业务执行系统。它的第一层能力是多重调度，也就是先判断用户是在问知识、办任务还是已经超出业务范围，再决定应该进入销售、售后还是财务域；第二层能力是多执行器协同，高风险任务由显式状态机接管，普通任务由 Function Calling Agent 执行，模型能力不足时还能回退到 ReAct；第三层能力是知识中台化，通过混合检索、缓存、预取、真实知识源和统一回答模板，把产品介绍、价格套餐、商品目录、实体说明、订单物流、退款账务等问题都纳入同一套 RAG 能力范围。这样做的结果是，系统既有大模型的理解和生成能力，又保留了工程侧对流程、权限、边界和调试性的控制。


## 2. 当前最重要的结构结论

先记住这几个结论，再去看代码会非常清楚。

1. `voice_optimized_rag/core/memory_router.py` 仍然是知识问答主链的中央协调器。
2. `voice_optimized_rag/dialogue/task_state_machine.py` 现在先接管退款、取消订单、修改地址三类高价值任务。
3. `voice_optimized_rag/agent/function_calling_agent.py` 现在是显式状态机未命中时的业务执行主链。
4. `voice_optimized_rag/agent/skill_registry.py` 现在是销售、售后、财务技能定义的正式注册表。
5. `voice_optimized_rag/dialogue/intent_router.py` 现在不再做 `chitchat` 分类，而是三态：
   - `knowledge`
   - `task`
   - `out_of_scope`
6. `voice_optimized_rag/dialogue/business_scope.py` 现在不只是拒答 prompt 工具，而是统一承载销售结构化回答和实体说明模板。
7. `examples/business_cli.py` 是本地无语音业务测试入口，也是当前最佳调试入口。
8. `voice_optimized_rag/llm/tracing.py` 提供共享的 LLM 调用轨迹包装器，CLI 已接入。
9. `web/server.py` 已经切到纯业务语义，不再保留闲聊分支。
10. `voice_optimized_rag/mcp_server.py` 和 `examples/business_mcp_server.py` 负责对外提供 MCP 能力。


## 3. 当前目录应该怎么理解

现在可以把仓库分成七层。

### 第 1 层：交付入口层

- `examples/customer_service_demo.py`
- `examples/business_cli.py`
- `examples/business_mcp_server.py`
- `examples/voice_demo.py`
- `web/server.py`

这一层解决“系统最终如何被使用”。

### 第 2 层：系统协调层

- `voice_optimized_rag/core/memory_router.py`
- `voice_optimized_rag/core/fast_talker.py`
- `voice_optimized_rag/core/slow_thinker.py`
- `voice_optimized_rag/core/conversation_stream.py`
- `voice_optimized_rag/core/semantic_cache.py`

这一层解决“RAG 主链如何同时追求低延迟和高召回”。

### 第 3 层：会话与策略层

- `voice_optimized_rag/dialogue/session.py`
- `voice_optimized_rag/dialogue/intent_router.py`
- `voice_optimized_rag/dialogue/domain_router.py`
- `voice_optimized_rag/dialogue/business_scope.py`
- `voice_optimized_rag/dialogue/follow_up.py`
- `voice_optimized_rag/dialogue/task_slots.py`
- `voice_optimized_rag/dialogue/task_state_machine.py`
- `voice_optimized_rag/dialogue/memory_manager.py`
- `voice_optimized_rag/dialogue/emotion_detector.py`
- `voice_optimized_rag/dialogue/transfer_policy.py`

这一层解决“请求属于什么业务、会话当前状态是什么、当前 workflow 在哪一 stage、哪些输入应被当作任务补槽或销售知识续接、应不应该拒答或转人工”。

### 第 4 层：执行层

- `voice_optimized_rag/agent/base_tool.py`
- `voice_optimized_rag/agent/function_calling_agent.py`
- `voice_optimized_rag/agent/react_agent.py`
- `voice_optimized_rag/agent/skill_registry.py`
- `voice_optimized_rag/agent/domain_agent.py`
- `voice_optimized_rag/agent/permission_guard.py`
- `voice_optimized_rag/agent/tools/`
- `voice_optimized_rag/mcp_server.py`

这一层解决“如何安全、结构化地调用业务工具”。

### 第 5 层：检索与知识库层

- `voice_optimized_rag/retrieval/vector_store.py`
- `voice_optimized_rag/retrieval/qdrant_store.py`
- `voice_optimized_rag/retrieval/inverted_index.py`
- `voice_optimized_rag/retrieval/hybrid_retriever.py`
- `voice_optimized_rag/retrieval/kb_manager.py`

这一层解决“知识如何切块、存储、双路召回、融合和重排”。

### 第 6 层：模型与 provider 层

- `voice_optimized_rag/llm/base.py`
- `voice_optimized_rag/llm/tracing.py`
- `voice_optimized_rag/llm/openai_provider.py`
- `voice_optimized_rag/llm/siliconflow_provider.py`
- `voice_optimized_rag/llm/ollama_provider.py`
- `voice_optimized_rag/voice/stt.py`
- `voice_optimized_rag/voice/tts.py`

这一层解决“上层抽象最终由哪个模型或服务实例承载”。

### 第 7 层：配置、质检、日志与指标层

- `voice_optimized_rag/config.py`
- `voice_optimized_rag/utils/auto_qa.py`
- `voice_optimized_rag/utils/session_logger.py`
- `voice_optimized_rag/utils/metrics.py`

这一层解决“参数如何统一、回复如何校验、会话如何留痕”。


## 4. 从上到下看：当前主链怎么跑

### 4.1 纯文本业务主链：`examples/customer_service_demo.py`

这是当前最适合理解“业务编排”的入口。

完整步骤如下：

1. 用户输入业务文本。
2. 系统创建或复用 `SessionContext`，递增轮次。
3. `IntentRouter` 做纯业务三态判定：
   - `knowledge`
   - `task`
   - `out_of_scope`
4. 如果不是 `out_of_scope`，继续由 `DomainRouter` 路由到：
   - `sales`
   - `after_sales`
   - `finance`
5. `TransferPolicy` 判断是否需要转人工。
6. 根据意图分流：
   - `task` -> 先走 `BusinessTaskStateMachine`，未命中再走对应业务域的 `FunctionCallingAgent`
   - `knowledge` -> `MemoryRouter.query()`（CLI trace 模式下会走 `query_with_trace()`）
   - `out_of_scope` -> 固定业务范围引导语
7. `AutoQA` 做质检。
8. `MemoryManager` 写入短期记忆和摘要。
9. `SessionLogger` 记日志。

这里最关键的变化是：

- 不再有“闲聊走 LLM 直答”这条链。
- 任何非业务请求都会明确落到 `out_of_scope`。
- 多轮任务不再只靠模型“自己决定下一步”，退款 / 取消订单 / 修改地址先走显式状态机。
- 销售知识问题已经补上统一的结构化回答模板，像价格、套餐、试用、折扣、商品目录、实体说明、短型号查询都会被共享 prompt 层显式处理。


### 4.2 本地纯业务 CLI：`examples/business_cli.py`

这是新增的本地测试入口，专门给你在没有语音链路的情况下做业务联调。

它的设计目标非常明确：

1. 自动尝试连接本地 Ollama。
2. 如果本地服务没起，会尝试执行 `ollama serve`。
3. LLM provider 固定走 `ollama`。
4. embedding provider 也固定走 `ollama`。
5. `stt_provider` 和 `tts_provider` 明确设成 `none`。
6. 支持 `quit / exit / 退出` 退出。
7. 支持 `--trace`，并且运行时可用 `trace on / trace off` 动态开关调试轨迹。

所以它是一条真正的“本地文本业务链”，而不是把语音链关掉之后凑出来的 demo。

当前 CLI trace 会同时输出两层信息：

1. 模块决策轨迹
   - `intent_router`
   - `domain_router`
   - `task_state_machine`
   - `transfer_policy`
   - `knowledge_rag`
   - `auto_qa`
2. LLM 调用轨迹
   - `generate / stream / complete_with_tools`
   - provider / model
   - prompt 长度 / context 长度
   - latency
   - response 摘要
   - function calling 的 tool 列表与 tool call 摘要

要特别注意：

> 当前 CLI 打印的是“模块决策轨迹 + LLM 调用轨迹”，不是模型私有原始思维链。这种输出更稳定，也更适合做工程调试。


### 4.3 Web 主链：`web/server.py`

`web/server.py` 现在也已经切到纯业务语义。

当前 `/ws/chat` 主链是：

1. 浏览器通过 WebSocket 发送文字或音频。
2. 如果是音频并且云端 STT 可用，先做语音转写。
3. `EmotionDetector` 更新会话情绪。
4. `IntentRouter` 分类为 `knowledge` / `task` / `out_of_scope`。
5. 如果不是 `out_of_scope`，再由 `DomainRouter` 判定业务域。
6. `TransferPolicy` 判断是否需要转人工。
7. 根据意图分流：
   - `task` -> 先走 `task_state_machine.handle(...)`，未命中再走 `domain_agents[session.current_domain].execute(...)`
   - `knowledge` -> `llm.stream(build_business_answer_prompt(...))`
   - `out_of_scope` -> 直接返回固定拒答
8. 文本被切片后送入 TTS 队列，边合成边推送音频。
9. 最终经过 `AutoQA` 和 `SessionLogger`。

这里最重要的事实是：

- `web/server.py` 已经接入 `DomainRouter`。
- `web/server.py` 已经切到多域业务 agent，而不是单个 `ReactAgent`。
- `web/server.py` 也不再保留闲聊设计。


### 4.4 MCP 对外互操作链：`voice_optimized_rag/mcp_server.py`

MCP 现在已经是仓库里的现有能力，而不是规划项。

当前 MCP server 暴露了四类能力：

1. `route_business_request`
   - 对外提供纯业务意图和业务域路由结果。
2. `query_business_knowledge`
   - 对外提供基于 `MemoryRouter` 的知识查询。
3. `execute_business_task`
   - 对外提供“显式状态机优先，FunctionCallingAgent 兜底”的业务任务执行。
4. `handle_business_request`
   - 对外提供统一业务处理入口。

同时还暴露了两个资源或辅助能力：

1. `business://skills`
2. `business://config`

以及一个面向外部客户端的 prompt：

1. `business_triage_prompt`

因此，MCP 在当前项目里的定位已经非常明确：

> 它不是内部工具调用机制本身，而是把已有的业务中台能力标准化暴露给外部 Agent 和客户端的互操作层。


### 4.5 知识主链：`MemoryRouter`

知识问答主链没有被这次“业务化收敛”推翻，它仍然是整套知识能力的中央协调器。

执行过程仍然是：

1. 发布 `USER_UTTERANCE` 事件。
2. `FastTalker` 处理当前问题。
3. `SlowThinker` 在后台预测后续问题并预取。
4. `FastTalker` 优先读 `SemanticCache`。
5. 未命中再走 `HybridRetriever`。
6. LLM 基于上下文输出当前回复。
7. 发布 `AGENT_RESPONSE` 事件写回对话流。

所以当前项目并不是“只剩下 function calling”，而是：

- 知识主链仍由 RAG 中台承担。
- 操作主链已经变成“显式任务状态机优先，原生 function calling 兜底”。


### 4.6 语音 RAG 演示链：`examples/voice_demo.py`

这条链仍然主要是“语音化的 RAG 演示”。

它的重点是：

1. 麦克风输入。
2. STT 转文本。
3. 文本进入 `MemoryRouter.query_stream()`。
4. 回答文本进入 TTS。
5. 音频播放。

它没有像业务入口那样完整接入意图路由、业务域技能注册和本地 CLI 运行逻辑，所以它更适合看“语音闭环”，不适合看“业务调度”。


## 5. 执行层现在怎么抽象

这是这次变化最大的部分。

### 5.1 BaseTool 不再只是 prompt 描述

以前 `BaseTool` 主要提供两件事：

1. prompt 描述文本
2. `execute()` 抽象接口

现在它新增了两种导出能力：

1. `to_json_schema()`
2. `to_function_schema()`

这意味着当前工具层已经不仅能给 ReAct 用，也能直接给原生 function calling 用。

这是整个执行层升级的第一块地基。


### 5.2 LLMProvider 现在支持原生工具调用协议

`voice_optimized_rag/llm/base.py` 现在新增了：

- `ToolCall`
- `ToolCallingResponse`
- `supports_function_calling`
- `complete_with_tools()`

这使得上层 agent 不再需要知道底层到底是 OpenAI、SiliconFlow 还是 Ollama，只需要调统一的原生函数调用接口。


### 5.3 哪些 provider 已经接入原生 Function Calling

当前已经接入原生工具调用的 provider 有：

- `OpenAIProvider`
- `SiliconFlowProvider`
- `OllamaProvider`

这点很关键，因为它带来了两个直接收益：

1. 云端模型链可以稳定用结构化工具调用。
2. 本地 CLI 也可以用本地 Ollama 模型直接做原生 function calling。

换句话说，这次不是“只给 OpenAI 单独加 function calling”，而是把 provider 抽象层整体补齐了。


### 5.4 显式任务状态机先补上 deterministic 业务流程

`voice_optimized_rag/dialogue/task_state_machine.py` 是最近这一轮非常关键的新抽象。

它当前显式接管三类高价值任务：

1. `refund`
2. `cancel_order`
3. `update_address`

执行顺序是：

1. 根据用户输入检测 workflow。
2. 结合 `task_slots.py` 同步 `order_id / phone / reason / amount / new_address`。
3. 如果必填信息不完整，直接用固定话术追问。
4. 信息齐全后先走 `PermissionGuard`。
5. 确认通过后再执行底层工具。
6. 完成后清空 workflow 相关 slots，并把 `task_status` 复位到 `idle`。

这带来的本质变化是：

- 退款、取消订单、改地址不再主要靠模型自己推断下一步。
- 多轮续接条件绑定 `active_workflow`，不会因为上一轮是 task 就把下一条短输入误判成补槽。
- 工作流结束后会清掉旧订单号和旧原因，避免后续任务被脏槽位污染。


### 5.5 FunctionCallingAgent 的职责

`voice_optimized_rag/agent/function_calling_agent.py` 现在是显式状态机未命中时的主业务执行器。

但要注意：它现在已经不是“所有 task 的第一入口”。

当前真实顺序是：

> 显式任务状态机优先；状态机未命中的 task，再交给 `FunctionCallingAgent`。

它的执行逻辑是：

1. 判断当前 provider 是否支持原生 function calling。
2. 如果支持：
   - 读取工具 schema。
   - 调 `llm.complete_with_tools()`。
   - 获取结构化 `tool_calls`。
   - 先过 `PermissionGuard`。
   - 再执行工具。
   - 把结果写回 scratchpad，继续下一轮。
3. 如果不支持：
   - 自动回退到 `ReactAgent`。

所以它不是彻底抛弃 ReAct，而是：

> 原生 Function Calling 优先，ReAct 作为 provider 兼容回退。

这比单纯保留 ReAct 更稳，也比硬切成“只有 OpenAI 能跑”的方案更实用。


### 5.6 SkillRegistry 现在是正式抽象，不再只是 prompt 工厂

`voice_optimized_rag/agent/skill_registry.py` 现在提供了正式的 `SkillRegistry`。

每个 `SkillSpec` 统一定义：

- `skill_id`
- `domain`
- `agent_name`
- `responsibility`
- `scope_rules`
- `tool_factories`

当前默认注册表是 `BUSINESS_SKILL_REGISTRY`，里面正式承载三类技能：

1. 销售 skill
2. 售后 skill
3. 财务 skill

这样做的本质是把“领域化约束”从过去的 `DomainAgentSpec` 升级成正式注册模型。


### 5.7 domain_agent.py 现在是什么角色

`voice_optimized_rag/agent/domain_agent.py` 现在主要是兼容层。

它保留了：

- `create_domain_agents()`
- `DOMAIN_AGENT_SPECS`

但底层实际已经切到 `SkillRegistry`。

这样做的好处是：

- 旧调用方不用立刻全部重写。
- 新架构已经落到正式注册表。

这是一种比较稳妥的迁移方式。


## 6. 纯业务路由现在怎么理解

### 6.1 为什么删除 chitchat

因为你的目标已经明确不是“陪聊助手”，而是“纯业务客服系统”。

继续保留 `chitchat` 会带来两个问题：

1. 业务入口会被无关请求稀释。
2. 系统会默认尝试给非业务内容一个开放式大模型回答，这和业务系统的边界不一致。


### 6.2 当前替代方案是什么

当前不是把 `chitchat` 换成另一个闲聊分支，而是换成：

- `out_of_scope`

语义上它表示：

- 不是销售问题
- 不是售后问题
- 不是财务问题
- 系统应该拒答并引导用户回到业务问题


### 6.3 当前意图层的三态

当前 `IntentType` 只有三种：

1. `knowledge`
2. `task`
3. `out_of_scope`

这意味着当前系统的主业务判断逻辑已经变成：

1. 能不能回答知识。
2. 能不能执行操作。
3. 如果都不是，就明确拒答。


### 6.4 业务范围拒答是怎么统一的

`voice_optimized_rag/dialogue/business_scope.py` 统一提供：

- `OUT_OF_SCOPE_RESPONSE`
- `build_business_answer_prompt()`

这让各个入口层可以统一表达同一件事：

- 非业务请求不要闲聊，不要继续扩展，不要安抚性陪聊。
- 直接把用户引导回销售、售后、财务相关问题。

但这层现在已经不只是“拒答 prompt 工具”。

它还统一承载了五类业务知识模板：

1. 结构化销售价格模板
   - 价格总览
   - 套餐对比
   - 试用政策
   - 折扣与优惠
2. 产品介绍模板
3. 商品目录 / 产品线模板
4. 实体说明模板
   - 例如 `Zendesk是什么`
   - `什么是Zendesk`
5. 短型号 / 编号 / 模块代号模板
   - 例如 `nove`
   - `nvme`

这意味着当前共享 prompt 层的职责已经变成：

- 统一业务范围边界
- 统一结构化销售知识回答格式
- 统一“商品目录 / 外部平台 / 实体说明 / 短代号查询”这些高频销售知识问法


## 7. RAG 部分怎么做到混合检索、双路召回、倒排、rerank

这一部分没有被本次“纯业务化”和“function calling 化”影响，仍然是当前仓库最成熟的中台能力。

### 7.1 入库路径

1. 文档读取。
2. 文本切块。
3. embedding 生成。
4. 向量写入 FAISS 或 Qdrant。
5. 更新 `document_version`。


### 7.2 双路召回

`HybridRetriever.search()` 同时走：

1. Dense 检索
2. Sparse 倒排 BM25 检索

这就是当前的双路召回。


### 7.3 倒排索引

`inverted_index.py` 通过：

- 中文 2-gram / 3-gram
- 英文单词 token
- BM25 打分

构建轻量稀疏检索能力。


### 7.4 融合

融合策略是 RRF。

原因是：

- dense 和 sparse 的分值尺度不同。
- RRF 对 rank 稳定、实现轻量。


### 7.5 rerank

当前 rerank 是启发式精排，不是额外 cross-encoder。

它综合：

- fused score
- dense score
- sparse score
- token 覆盖率
- phrase boost

这是当前仓库阶段性最实用的方案。


### 7.6 查询扩展现在不只是价格词，还覆盖商品目录和产品代号

`hybrid_retriever.py` 里的 `expand_query_text()` 已经补上了更多中文到英文的业务别名映射。

除了价格、套餐、优惠、退款、发票、物流之外，现在还覆盖：

- `商品`
- `产品`
- `目录`
- `模块`
- `型号`
- `编号`

这样做的目的很直接：

> 当前知识库很多内容仍然是英文摘要或英文原文，中文业务问法如果不做稳定别名扩展，召回质量会明显下降。


### 7.7 真实公开业务数据已经正式进入知识库

当前知识库已经不再只包含内部的 NovaCRM 示例资料，还新增了：

- `knowledge_base/real_world/13_pipedrive_sales_pricing.txt`
- `knowledge_base/real_world/14_zendesk_after_sales_pricing.txt`
- `knowledge_base/real_world/15_stripe_billing_finance_summary.txt`
- `knowledge_base/real_world/16_zoho_books_finance_summary.txt`

这些文件的定位不是“原网页拷贝”，而是：

- 按销售 / 售后 / 财务场景整理过的摘要型知识源
- 用来做真实业务问法测试
- 让系统能回答外部平台、竞品、参考 SaaS 的价格、试用、购买和功能差异

因此现在像：

- `Zendesk是什么`
- `Zendesk 年付有折扣吗`
- `Pipedrive 有免费试用吗`
- `Stripe Billing 怎么收费`

都已经属于当前 RAG 可覆盖的问题空间。


### 7.8 入库链路已经补上去重和脚本修复

这轮还有一个非常实用但容易被忽略的变化：

1. `vector_store.py` 已经按 `source + chunk_index` 去重。
2. `MemoryRouter.ingest_directory()` 会返回真实新增块数。
3. `examples/ingest_documents.py` 的目录参数 bug 已修复。

这意味着：

- 重复 ingest 不会再把索引越灌越脏。
- 加入新的真实公开数据后，可以稳定重复入库验证。
- “召回变差”不再像之前那样经常是由索引污染引起的。


## 8. 本地语音和本地 CLI 的关系

### 8.1 本地 ASR/TTS 仍然停用

当前本地语音实现仍然是停用状态：

- `sensevoice_stt.py` 为停用 stub
- `cosyvoice.py` 为停用 stub
- 工厂层直接拒绝本地 provider 分支

这条结论没有变化。


### 8.2 本地 CLI 不是恢复本地语音，而是走本地文本 LLM

新增的 CLI 和本地语音完全不是一回事。

它做的是：

- 本地 Ollama LLM
- 本地 Ollama Embedding
- 文本输入
- 文本输出

它没有调用任何语音 provider。

所以本地 CLI 的意义是：

> 让你能在“本地模型 + 纯业务文本”的条件下联调整个业务链，而不是把停用的本地语音链重新拉回来。


### 8.3 本地 CLI 现在还是最重要的调试入口

现在 `business_cli.py` 不只是“本地能跑起来”的入口，还承担调试职责。

它已经能输出：

1. 模块决策轨迹
   - intent 判定
   - domain 判定
   - state machine 是否接管
   - transfer policy 结果
   - knowledge RAG 走缓存还是走检索
   - 命中的来源文件
   - auto_qa 是否清洗回复
2. LLM 调用轨迹
   - 调用了几次 `generate / stream / complete_with_tools`
   - 哪个 provider / model
   - prompt 和 context 大小
   - latency
   - response 预览
   - function calling 的 tool 列表与 tool call 摘要

所以当前 CLI 在项目里的角色，已经从“文本版 demo”升级成：

> 本地业务链的联调入口 + 当前最直接的解释性调试入口。


## 9. 当前哪些地方已经统一，哪些还没完全统一

### 9.1 已经基本统一的部分

- `BaseTool` 现在同时服务 ReAct 和原生 Function Calling。
- `LLMProvider` 现在具备统一的原生函数调用抽象。
- 退款 / 取消订单 / 修改地址已经先统一到显式任务状态机，再回退到 Function Calling Agent。
- 销售、售后、财务已经正式收敛到 `SkillRegistry`。
- 主业务入口已经切到 `out_of_scope` 语义，不再保留 chitchat 分支。
- 销售知识回答已经统一收敛到 `business_scope.py` 的共享模板层。
- 外部平台 / 竞品 / 参考 SaaS 资料已经正式进入业务知识范围，而不是“问了就拒答”。
- 本地无语音业务测试入口已经补齐。
- 本地 CLI 的模块决策轨迹和 LLM 调用轨迹已经补齐。
- MCP 互操作层已经落到可运行代码，而不是停留在设计阶段。
- 本地语音停用仍然统一由配置默认值、工厂层和 stub 三个控制点管理。


### 9.2 仍然没有完全统一的部分

- `voice_demo.py` 仍然主要是语音化 RAG 演示，不是纯业务入口。
- `web/server.py` 的知识问答仍然是 `MemoryManager + LLM` 轻路径，没有直接复用 `MemoryRouter`。
- Web 和 MCP 目前还没有像 CLI 一样直接暴露完整的模块轨迹和 LLM 调用轨迹。
- 商品目录 / SKU / 别名映射还没有被整理成一份更正式的结构化产品目录表，短代号查询目前主要依赖 prompt 约束和 alias 扩展。
- ReAct 仍然保留在代码里作为兼容回退，而不是被彻底移除。


### 9.3 这说明什么

说明当前项目已经出现了比较清晰的主干：

- 业务操作优先走显式任务状态机，再回退到原生 function calling。
- 业务知识优先走 RAG。
- 非业务请求统一拒答。

剩下的差异主要在“不同入口对中台能力的复用程度”上，而不在“核心抽象有没有形成”上。


## 10. 如果后面继续开发，最合理的演进方向

### 10.1 让 Web 知识链直接复用 MemoryRouter

现在 Web 端知识链仍然偏轻量。

如果后面继续收敛，最自然的方向是让 `web/server.py` 直接复用 `MemoryRouter`，这样就能把：

- `SlowThinker` 预取
- `FastTalker` 缓存
- `HybridRetriever` 混合检索

完整带到 Web 端。


### 10.2 把商品目录 / SKU / 别名做成正式知识源

现在像 `nove`、`nvme` 这类短输入，系统已经不会再直接拒答，但能否精准命中仍然受限于知识源本身。

如果后面要继续提升这条链，最自然的方向是补一份结构化产品目录文档，明确写出：

- 产品主名称
- 模块名称
- 常见别名
- 英文简称
- 短代号
- 适用场景

这样“短代号 -> 精准产品说明”就会从 prompt 技巧升级成稳定知识能力。


### 10.3 继续扩展 SkillRegistry，而不是继续堆 if/else

如果后面要新增“发票开具 skill”“退款审核 skill”“安装预约 skill”，建议继续往 `SkillRegistry` 里加 `SkillSpec`，而不是把逻辑重新写回入口文件。


### 10.4 把 trace 能力继续接到 Web / MCP

当前最完整的调试体验在 CLI。

如果后面继续收敛，最自然的下一步是把：

- 模块决策轨迹
- LLM 调用轨迹

也接到 `web/server.py` 和 `mcp_server.py`，这样不同入口的解释性就能对齐。


### 10.5 MCP 现在的定位

当前 MCP 已经落地，但它的定位仍然应该保持清晰：

- MCP client 用来接外部系统能力。
- MCP server 用来暴露现有业务工具、业务路由和知识查询。

也就是说，MCP 现在已经存在于仓库里，但它仍然属于“跨系统互操作层”，而不是取代内部业务执行层本身。


## 11. 推荐的阅读顺序

如果你要系统性学习这套项目，建议按下面的顺序：

1. 先读 `config.py`，知道整体开关和 provider 选择方式。
2. 再读 `llm/base.py`、`llm/tracing.py` 和 `agent/base_tool.py`，理解 LLM 抽象、调用轨迹和工具 schema 接口。
3. 再读 `task_state_machine.py`、`task_slots.py`、`function_calling_agent.py`、`react_agent.py`、`skill_registry.py`，理解业务执行层为什么变得更 deterministic。
4. 再读 `session.py`、`follow_up.py`、`intent_router.py`、`domain_router.py`、`business_scope.py`，理解纯业务路由和共享回答模板。
5. 再读 `hybrid_retriever.py`、`memory_router.py`，并顺手看 `knowledge_base/real_world/`，理解知识主链和真实公开数据是怎么接进来的。
6. 再读 `mcp_server.py`，理解内部能力是如何被包装成对外互操作接口的。
7. 最后回到 `customer_service_demo.py`、`business_cli.py`、`business_mcp_server.py`、`web/server.py`，看这些中台抽象如何被拼成不同入口链。

这样读的好处是：

- 你先掌握骨架，再看入口。
- 你会知道每个入口到底复用了什么，而不是被流程代码带着跑。


## 12. 一句话总结

当前仓库最值得抓住的，不是“接了多少模型”，而是它已经形成了更清晰的业务骨架：

- 用 `IntentRouter` 把请求限制在知识、任务、超范围三态。
- 用 `DomainRouter` 把业务继续细分到销售、售后、财务。
- 用 `task_state_machine.py` 先把高价值任务流程做成 deterministic 状态机。
- 用 `BaseTool + FunctionCallingAgent + SkillRegistry` 形成原生函数调用执行层。
- 用 `MemoryRouter + HybridRetriever` 形成混合检索知识中台。
- 用 `business_scope.py` 把价格模板、产品介绍、商品目录、实体说明、短型号查询统一收敛到共享回答模板层。
- 用 `business_cli.py` 提供本地无语音联调入口和最强调试入口。
- 用 `mcp_server.py` 把已有业务能力标准化暴露给外部系统。

后续开发如果继续沿着这套骨架收敛，复杂度会越来越稳定；如果重新回到入口层硬写分支和 prompt，系统会再次变成难维护的流程脚本。