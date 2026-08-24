# RAG 重构 + 工具调用自主决策 + 观测治理 — 技术方案（决策稿）

## Context

对现有智能客服的检索与编排做一次架构演进，四项诉求：

1. **RAG 换底**：检索层改用 LlamaIndex 完整接管。
2. **向量库替换**：ChromaDB → Milvus（docker-compose 独立 standalone）。
3. **工具自主决策**：RAG 查询不再写死在流程里，包装成 function calling，由 LLM 决定是否调用；最终扩展为**全量 7 个工具**交 LLM 决策。
4. **观测治理**：做请求分析时能监控 tool 调用是否合理。

**已确认的决策**（2026-08-24 评审确认）：
- LlamaIndex **完整接管检索层**（保留 `Retriever.search()` 外观与 Redis 缓存）。
- Milvus 用 **docker-compose 独立 standalone**（milvus + etcd + minio）。
- **全量 7 个工具**交 LLM 决策（业务副作用工具经安全闸收敛，见 §5）。
- **`run_agent` 整体重构为 LangGraph 顶层图**。

技术栈约束：LangGraph 0.2.60 / langchain-core 0.3.25 / FastAPI / DeepSeek（OpenAI 兼容接口）。

---

## 1. 现状梳理（关键事实）

| 领域 | 现状 | 与本方案的关联 |
|---|---|---|
| 顶层编排 | `run_agent` 六阶段命令式流水线（预处理→意图→装配→状态推进→动作→响应），**不在 LangGraph 图内** | 需整体图化 |
| LangGraph | 仅驱动 3 个业务状态机（退货/退款/投诉），`BaseStateMachine.step()` 每轮输入推进一节点 | 子图复用，节点逻辑不变 |
| function calling | `function_calling/` 已有 7 个工具的 schema + handler（`registry.py`/`executor.py`/`tools/`），但 **`TOOL_SCHEMAS`、`executor.execute()` 零调用点，未接入任何 LLM 工具循环** | 骨架在，缺"LLM 决策→执行→回灌"链路 |
| RAG | `retriever.search()` → bge-small-zh embedding → `ChromaVectorStore` → Top-10 → **按分排序取 Top-3（非真重排）** → score≥0.3 过滤 → Redis 精确缓存 600s | LlamaIndex 替换内部，外观保留 |
| 知识库一致性 | MySQL `knowledge_docs` 是 source of truth，Chroma 只是派生向量索引；`kb_store` 增量补偿 + 全量对账 | **换库零数据迁移**，`sync_full()` 重建即可 |
| 向量库抽象 | `IVectorStore` 已抽象，`app/rag/__init__.py` 工厂，`config.vector_store` 已支持 `milvus` 值 | 预留路径已就位 |
| LLM 客户端 | `DeepSeekGateway.chat()` 不传 `tools`/`tool_choice` 参数 | 需加透传，改动小 |
| 工具观测 | `tool_call_event` 仅 SSE 外显（决策 #8：不强制 LLM function calling），**不落库、无合理性判定** | 需新增观测 + 护栏 |

**决策 #8 背景**：现设计刻意"不强制 LLM function calling"，工具由流水线/状态机确定性驱动、观测式外显，保证评测确定性与成本。本次改造是对该决策的一次主动演进——从"确定性"转向"LLM 自主 + 护栏兜底"。

---

## 2. 可行性评估

- **Milvus 替换**：最顺。走 LlamaIndex 的 `MilvusVectorStore`，MySQL 是源所以数据零迁移，`kb_store` 一致性逻辑原样复用。
- **RAG 包装成 function calling**：可行，但有一个架构级取舍——决策 prompt 必须**默认倾向检索**（防 LLM 不检索裸答编造政策），且决策轮是额外一次非流式 LLM 往返（约 +1~3s + token），与"确定性优先"的既有方向冲突，需实测。
- **监控 tool 合理性**：设计为"实时护栏 + 落库 + 离线分析"三层，可独立于检索改造先行落地（本方案放 P4/P5）。

---

## 3. 目标架构

```
run_agent 重构为 LangGraph 顶层图（每轮用户输入跑一次）
START
 → preprocess            # 注入检测（现状平移）
 → intent_classify       # LLM 六分类 + 槽位提取 + 状态机"推进/切换"判断（保留）
 → agent_loop ───────────  # 核心：LLM + 全量工具决策（非流式，≤3 轮）
      │  tools 全集: query_order / list_user_orders / search_policy
      │              check_return_eligibility / create_return_order /
      │              create_refund_order / create_complaint
      │  ↓ 实时护栏 ToolGuardrail（合理性校验，allow/reject/override）
      │  ↓ tool_call_log 落库（每个决策点一条）
      ├─ 订单类 → order_answer    (流式生成)
      ├─ 政策类 → policy_answer   (search_policy 结果注入 + 流式生成)
      ├─ 闲聊类 → chitchat_answer (流式生成)
      └─ 业务类 → business_flow  (退货/退款/投诉 LangGraph 子图，挂现有状态机)
 → finalize             # usage 聚合 + answer 补发 + 状态落库
END
```

**设计要点**：
- 决策与生成**分两段**：`agent_loop` 非流式只做工具决策与上下文收集；最终答案一律走专用节点流式生成。保住 `answer.delta` SSE 契约、usage 聚合、`_rule_engine_fallback` 熔断降级（装饰器靠异常冒泡 + tracked_emit 已流部分拼接，图内依然生效）。
- 状态机"推进/切换/快照"逻辑平移进图的意图路由边，行为等价。
- `agent_loop` 的每个非流式调用 usage 计入本轮聚合（现状意图分类已这么做）。

---

## 4. 分阶段落地计划

**阶段节奏**：P1 与 P2 可并行；P2 完成后 P3→P5 串行。每阶段契约测试全绿才进下一阶段。

### P1：RAG 换底（LlamaIndex + Milvus）

- LlamaIndex 完整接管检索层：
  - Embedding：`HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")`，**维度 512 不变**。
  - 存储：`MilvusVectorStore`（pymilvus，`uri=http://milvus:19530`，collection `knowledge`）。
  - 检索：`VectorIndexRetriever(similarity_top_k=10)`。
  - 重排：**新增 bge-reranker 真交叉编码重排**（`SentenceTransformerRerank`，top_n=3）——修掉现状"按分排序冒充重排"。
  - 阈值过滤 + Redis 缓存：保留在 `Retriever` 外观内。
- `Retriever.search()` 签名、`SCORE_THRESHOLD`、缓存行为**不变**，调用方（`search_policy` handler、orchestrator）无感。
- `kb_store.py`：`rebuild_source` 改为写 LlamaIndex 索引；upsert/delete/对账语义不变。
- docker-compose：新增 `milvus`（+ `etcd` + `minio`）服务，替换 `chroma`；配置项 `VECTOR_STORE=milvus`。

**契约影响**：无对外变化（事件名/接口不变）。
**测试/验收**：`test_rag.py` 适配新实现；检索结果与现状抽样比对（Top-N 召回一致率 ≥ 阈值）；`test_knowledge.py` 全绿。

**P1.5 冒烟验证记录（2026-08-24）**——实机校准出的关键口径，写死以免后人再踩：

1. **Score 口径**：Milvus COSINE 的 `hit.distance` 即余弦相似度本身（同向量=1.0，实测），**不是 `1-cos`**。直接作相似度返回，与 Chroma 的 `(1 - cosine_distance)` 同口径，0.3 阈值语义一致。曾在 `1.0-distance` 上反转过分数（相关 0.31 vs 不相关 0.42），probe 校准后修正。
2. **A/B 一致性**：同批 8 文档 + 3 query，Milvus 与 Chroma(embedded) 的 Top-5 分数/排序**逐位一致**（如退货 query 首三位 0.694/0.687/0.610），向量检索数学等价确认。
3. **踩过的坑**（已修复并留注释）：
   - `get_collection_stats()['row_count']` 在 insert 后、flush 前 stale 为 0 → 用 `count(*)` + `consistency_level="Strong"` 即时准确。
   - Milvus store 的 `query()` 需要 `VectorStoreQuery`（`QueryBundle` 无 `.mode` 会 TypeError）。
   - `TextNode` 的 `ref_doc_id` 是**派生属性**，constructor 传入被忽略（`doc_id` 全变 "None"，`delete_by_source` 静默失效）→ 显式设置 `relationships[SOURCE]=RelatedNodeInfo(node_id=source)`。
   - metadata 无独立字段 → 从 `_node_content` JSON 还原。
   - `SentenceTransformerRerank` 参数名是 `top_n`，不是 `top_k`。
4. **Redis 降级**：`retriever` 对缓存 get/set 均 try/except，Redis 不可用时静默跳过缓存直接检索（本机 6379 端口被代理层拦截时验证通过）。
5. **Chroma 弃用陷阱**：chroma 服务器 1.4.4 已废弃 v1 API（410 Gone），chromadb 客户端 1.4.1 走 v1 → transitional 路径（`VECTOR_STORE=chroma`）必须用 embedded `PersistentClient`，不可再连 server mode。
6. **HF 网络**：bge-reranker-base 约 1.1GB，直连 huggingface.co 易卡死；hf-mirror.com 断点续传可用（range resume）。embedder 显式 `HF_ENDPOINT=https://huggingface.co` 可加载。
7. **FULL smoke（带真重排）验证**：rerank **只调顺序、保留原始 cosine score**（交叉编码 logit 不暴露，`reranker.py` 设计如此），故 0.3 阈值口径不变。实际价值：query「换货怎么申请」原始向量检索 top1 是 shipping「物流单号」（错），rerank 后 top1 修正为 aftersale「15 天内免费换新」。注意：重排后**序 ≠ 分数序**（logit 序 + cosine 分），消费方按返回顺序取 top1 即可。nonsense query 返回 1 条（0.320≥0.3），为预先存在的余弦阈值行为（原始检索顶分即 0.32，A/B 证明 Chroma 一致），非回归；真实知识库上需回归确认阈值。

### P2：顶层图化（纯重构）

把现有六阶段流水线**原样平移**成 LangGraph 图，行为完全等价：
- 节点：`preprocess` / `intent_classify`（含状态机推进/切换判断）/ 现有各分支（业务状态机、订单、政策、闲聊）/ `finalize`。
- 路由边按现状 `if/elif` 逻辑等价映射；硬编码路由先保留。
- 流式/usage/熔断语义全部保留（emit 回调在节点内透传）。

**契约影响**：无行为变化。
**测试/验收**：`test_orchestrator*.py`、`test_sse_contract.py`、`test_orchestrator_contract.py` 全绿即合入。

### P3：工具决策循环

- `DeepSeekGateway.chat()` 增加 `tools` / `tool_choice` 参数透传（OpenAI 兼容 payload）。
- `agent_loop` 手动工具循环（≤3 轮）：LLM+tools → 护栏校验 → 执行 → 回灌 tool result → 无 tool_call 则产最终路由。
- 路由变化：
  - `search_policy`：由 LLM 决定是否检索（默认倾向检索）。
  - `query_order` / `list_user_orders`：由 LLM 决定（现状 `_handle_order_status` 的兜底逻辑搬进护栏或决策 prompt）。
  - 业务动作工具：经安全闸收敛到 `business_flow` 子图。
- 决策 prompt 显式声明："只有高置信闲聊/无需政策依据才不调检索；宁可多检索不可裸答。"

**契约影响**：
- SSE tool_call 事件名 **`policy_search` → `search_policy`**：README 契约、`test_orchestrator_contract.py:247` 断言、评测 judge（如依赖事件名）需同步。
- usage 把 `agent_loop` 非流式调用计入聚合。
**测试/验收**：政策命中/未命中/闲聊/订单查询四场景契约测试；用现有评测样例实测是否扣分（若扣分回退闸门，评估"强制检索兜底"）。

### P4：实时护栏 ToolGuardrail

`agent_loop` 决策与执行之间插入规则校验（确定性、零成本），输出 `allow / reject / override`：

| 规则 | 动作 |
|---|---|
| `search_policy`：query < 4 字或纯问候 | reject（不必要） |
| `search_policy`：检索空结果 | 引导人工（现状已具备） |
| `query_order`：无 order_id 且上下文无订单 | override 为 `list_user_orders` / 追问 |
| `create_*` / `check_return_eligibility`：非状态机节点内 | reject，路由到子图 |
| 同轮同工具同参数重复调用 | dedupe（取首次结果） |
| 单轮 loop 工具调用次数 > 3 | 截断，强制出路由 |

护栏判定 + 理由写入 `tool_call_log`。
**测试/验收**：护栏单测，每规则一正一反。

### P5：观测与分析

- 新增 MySQL 表 `tool_call_log`（见 §6）。
- 新增 admin 只读接口 `GET /admin/tools/analysis`：按时段/工具/verdict 聚合（调用频次、拒绝率、空结果率、超时率）+ 抽样不合理调用明细。
- 可选：离线 LLM judge 批量打分"该调用是否合理"，输出 TOP-N 问题调用（服务于"请求分析"工作流）。
- 前端管理页（可选，后置）。

**测试/验收**：构造"合理/不合理"样本集，judge 命中率验收；接口契约测试。

---

## 5. 两个安全闸（评审确认，按推荐执行）

**闸 1：业务副作用工具不放进自由循环裸调。**
`check_return_eligibility` / `create_return_order` / `create_refund_order` / `create_complaint` 这 4 个是副作用工具。现状状态机保证它们在"订单已校验、资格已查、用户已确认"后才执行——这是既有测试钉死的确定性保证。LLM 自由循环裸调会引入：槽位缺失乱调、重复建单、未确认就执行。故：
- 这 4 个工具**仍注册给 LLM**（可见、可决策"走业务流"），但**执行被 LangGraph 收敛到 `business_flow` 子图**（节点逻辑原样保留）。
- `query_order` / `list_user_orders` / `search_policy` 三个只读工具在循环内自由执行。

**闸 2：决策与生成分两段。**
工具决策非流式，最终答案走专用节点流式。保住 `answer.delta` 契约、usage 聚合、熔断降级语义；避免单段 ReAct 导致 SSE 契约与评测整体重写。

---

## 6. 新增数据模型与接口

### `tool_call_log` 表（MySQL）

```sql
CREATE TABLE tool_call_log (
  id            BIGINT AUTO_INCREMENT PRIMARY KEY,
  session_id    VARCHAR(64) NOT NULL,
  user_id       INT NOT NULL,
  round_no      INT NOT NULL,              -- agent_loop 内第几轮
  tool_name     VARCHAR(64) NOT NULL,
  args_json     JSON,
  result_summary VARCHAR(512),             -- 结果摘要（前 N 字 + 命中数）
  latency_ms    INT,
  verdict       VARCHAR(16) NOT NULL,      -- allow / reject / override
  verdict_reason VARCHAR(255),
  llm_confidence FLOAT,                    -- LLM 决策置信度（如有）
  query_text    VARCHAR(500),              -- 本轮用户输入，便于请求分析
  created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_session (session_id),
  INDEX idx_tool_time (tool_name, created_at)
);
```

### admin 分析接口

`GET /admin/tools/analysis?from=&to=&tool=&verdict=`
返回：各工具调用频次、verdict 分布、空结果率、平均延迟 + 抽样明细（分页）。

---

## 7. 契约影响清单（全量盘点）

| 位置 | 变化 | 影响 |
|---|---|---|
| `test_orchestrator_contract.py:247` | `policy_search` → `search_policy` | 事件名断言 |
| README §契约 / SSE 文档 | tool_call 事件名同步 | 评测 judge 如依赖事件名需确认 |
| `DeepSeekGateway.chat()` | 新增 `tools`/`tool_choice` 透传 | 向后兼容（缺省不传） |
| `usage` 聚合 | `agent_loop` 非流式调用计入 | 口径不变，路径扩展 |
| `_rule_engine_fallback` | 图中异常冒泡 + tracked_emit | 语义不变，需图内验证 |
| 业务状态机 | 逻辑不变，挂载方式变为子图 | 状态机契约测试原样 |
| `_handle_policy` / `_handle_chitchat` | 平移为图节点，流式逻辑保留 | SSE 契约不变 |

---

## 8. 依赖与风险

| 风险 | 说明 | 应对 |
|---|---|---|
| Milvus standalone 资源 | milvus + etcd + minio 三容器，内存占用高 | compose 压资源上限；连接串留配置 |
| bge-reranker 模型体积 | base 版约 1GB | 评估镜像体积；HF 缓存 volume 已挂 |
| LlamaIndex × langchain-core 0.3.25 | langchain 集成滞后易版本冲突 | **不桥接** `LlamaIndexTool → to_langchain_tool`，`search_policy` 直接包 `Retriever.search()` |
| DeepSeek 工具决策评测表现 | 未知是否扣分 | 决策 prompt 默认倾向检索；实测，扣分则回退"强制检索兜底" |
| 全量工具 + 图重构同推 | 风险叠加 | 分阶段，每阶段契约测试全绿再进下一阶段 |
| 事件名变更 | 契约 + 评测联动 | 独立提交，显式告知 |

---

## 9. 验收与测试策略

- 核心逻辑（Service 层 / agent_loop 决策 / 护栏）覆盖正路径与关键异常路径；控制器、工具类不强制。
- 每阶段：编译（mvn 对应 pytest）+ 关联单测 + 契约测试全绿，验收由助手自动执行并汇报。
- P1 检索一致性抽样比对；P3 四场景契约测试 + 评测样例实测；P5 judge 命中率。

---

## 10. 关联文档

- [技术方案](solution.md)
- [API 文档](docs/API.md)
- [任务拆分](task.md)
