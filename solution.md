# 高并发 AI Agent 智能客服系统 — 技术方案

## Context

高并发 AI Agent 智能客服系统，面向大并发峰值场景设计，支持退换货政策查询、订单状态查询、订单退货、仅退款、投诉等场景。

- **前后端分离**：Vue 3 + TypeScript 前端独立部署，FastAPI 后端 API
- **Agent 侧**：Python（LangGraph + FastAPI），LLM 用 DeepSeek API（chat + reasoner 混用）
- **对接层**：ABC 接口 + Local 实现（直接操作 MySQL），未来可切换 Remote
- **基础设施**：Redis + MySQL + ChromaDB，无 Kafka
- **架构不妥协**：Agent 无状态，所有组件可水平扩展

---

## 1. 总体架构

```
                         ┌──────────────────────────────────────┐
  用户 ←── HTTPS ──────→ │  Nginx :80                          │
  浏览器                  │  ├─ /          → Vue 前端静态文件    │
                         │  └─ /api/*     → 后端（REST + SSE）   │
                         └──────────────┬───────────────────────┘
                                        │
                         ┌──────────────┴───────────────────────┐
                         │  Vue 3 前端 (独立部署)                │
                         │  ├─ LoginView / RegisterView         │
                         │  ├─ CustomerServiceView              │
                         │  │   ├─ ChatPanel + ProgressBar      │
                         │  │   ├─ OrderCard + ConfirmButton    │
                         │  └─ AdminView (知识库+种子数据, admin) │
                         └──────────────────────────────────────┘
                                        │ HTTP POST + SSE
                         ┌──────────────┴───────────────────────┐
                         │  FastAPI Agent API (多 worker)        │
                         │  ┌─────────────────────────────────┐ │
                         │  │ Agent Pipeline (6 阶段)          │ │
                         │  │  ├─ Preprocessor (注入检测)      │ │
                         │  │  ├─ IntentClassifier (chat)     │ │
                         │  │  ├─ StateMachine (LangGraph)    │ │
                         │  │  ├─ FunctionCalling → 对接层     │ │
                         │  │  ├─ DeepSeek Gateway            │ │
                         │  │  └─ ResponseGenerator (SSE)     │ │
                         │  └─────────────┬───────────────────┘ │
                         │                │                      │
                         │  ┌─────────────┴───────────────────┐ │
                         │  │ RAG Engine                      │ │
                         │  │  ├─ IVectorStore 接口             │ │
                         │  │  ├─ ChromaDB + Redis 缓存        │ │
                         │  │  └─ Retriever + Re-ranker       │ │
                         │  └─────────────────────────────────┘ │
                         │                │                      │
                         │  ┌─────────────┴───────────────────┐ │
                         │  │ 对接层 (Integration Layer)      │ │
                         │  │  ├─ IOrderService → LocalImpl    │ │
                         │  │  ├─ IReturnService → LocalImpl   │ │
                         │  │  ├─ IRefundService → LocalImpl   │ │
                         │  │  └─ IComplaintService → LocalImpl│ │
                         │  └─────────────────────────────────┘ │
                         └──────┬──────────────┬────────────────┘
                                │              │
                    ┌───────────┴───┐  ┌───────┴────────────┐
                    │ Redis         │  │ MySQL              │
                    │ - Session     │  │ - users + orders   │
                    │ - Cache       │  │ - return_orders    │
                    │ - Rate Limit  │  │ - refund_orders    │
                    └───────────────┘  │ - complaint_tickets│
                                       │ - conversation_hist│
                    ┌───────────────┐  └───────────────────┘
                    │ ChromaDB      │
                    │ (磁盘持久化)   │
                    └───────────────┘
```

**核心数据流（以退货为例）**：

```
用户: "我要退货，订单号 ORD-001"
  → IntentClassifier → RETURN_REQUEST
  → StateMachine → VERIFY_ORDER → IOrderService.query_order() → MySQL → 返回订单
  → CHECK_ELIGIBILITY → deepseek-reasoner → 判定可退
  → COLLECT_REASON → 用户输入"质量问题"
  → CONFIRM → 用户确认
  → EXECUTE → IReturnService.create_return() → status=APPROVED → MySQL INSERT
  → 回复: "退货单 RC-20240807-001 已创建，退款 ¥69.70 将在1-3个工作日内原路返回。"
```

---

## 2. Agent 核心设计

### 2.1 六阶段流水线

```
INPUT → [1.预处理] → [2.意图识别+切换] → [3.上下文装配] → [4.状态推进] → [5.动作执行] → [6.响应生成] → OUTPUT
```

#### 阶段 1：预处理
- **Prompt Injection 检测**（正则，不消耗 LLM 调用）
- 敏感词过滤、用户身份注入（JWT→user_id→Redis）、输入截断

#### 阶段 2：意图识别 + 切换判断（deepseek-chat, ≈200ms）

6 个意图类别：`POLICY_INQUIRY` / `ORDER_STATUS` / `RETURN_REQUEST` / `REFUND_REQUEST` / `COMPLAINT` / `CHITCHAT`

意图切换逻辑（复用阶段 2 分类结果）：
- 新意图 ≠ 当前 intent 且 confidence > 0.8 → 保存快照 `session:{sid}:snapshot:{intent}`（**按意图区分，每意图只存一份**）→ LangGraph 重新路由 → 回复"已保存退货进度，先帮您查 ORD-002 的物流。"
- 意图不变 → 继续当前状态机

#### 阶段 2b：JSON 校验与容错

LLM 返回的 JSON 可能被 markdown 包裹、尾随文本、字段缺失。统一处理：

```python
async def classify_intent(user_input, max_retries=2):
    for attempt in range(max_retries):
        raw = await gateway.call(intent_prompt(user_input), model=settings.deepseek_model_chat)
        try:
            json_str = _extract_json(raw["choices"][0]["message"]["content"])
            data = json.loads(json_str)
            result = IntentResult(
                intent=data.get("intent", "CHITCHAT"),
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                slots=data.get("slots", {}),
                missing_slots=data.get("missing_slots", []),
                summary=data.get("summary", ""),
            )
            if result.intent not in VALID_INTENTS: result.intent = "CHITCHAT"
            return result
        except (json.JSONDecodeError, KeyError, ValueError):
            if attempt == max_retries - 1: return _chitchat_fallback()
            intent_prompt += "\n[注意] 必须输出严格的 JSON 格式，不要包裹在 ```json``` 中。"

def _extract_json(text: str) -> str:
    m = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
    if m: return m.group(1)
    m = re.search(r'\{.*\}', text, re.DOTALL)
    return m.group(0) if m else text
```

#### 阶段 3：上下文装配
Redis 加载会话 → 有意图切换时恢复对应 intent 的快照

#### 阶段 4：状态推进（LangGraph StateMachine）

#### 阶段 5：动作执行
同步调用对接层接口（延迟 5-10ms）。CHITCHAT 跳过此阶段。

#### 阶段 6：响应生成（deepseek-chat, SSE 流式）
**SSE 同时推送阶段事件**，前端显示进度：

```
{"type":"status","stage":"intent","message":"正在理解您的问题..."}
{"type":"status","stage":"order_query","message":"正在查询订单 ORD-001..."}
{"type":"token","content":"您"}
{"type":"token","content":"的"}...
{"type":"done","intent":"RETURN_REQUEST"}
```

SSE 推送中检测 `request.is_disconnected()` 自动保存进度并取消生成，防止资源泄露。

### 2.2 槽位填充与追问

`missing_slots` 非空 → 不进入状态机，LLM 动态生成追问，同时附带用户最近订单列表。

```python
REQUIRED_SLOTS = {
    "POLICY_INQUIRY": [], "ORDER_STATUS": ["order_id"],
    "RETURN_REQUEST": ["order_id"], "REFUND_REQUEST": ["order_id"],
    "COMPLAINT": [],   # complaint_type 由 LLM 从输入提取
    "CHITCHAT": [],
}
```

### 2.3 状态机（LangGraph）

**退货（RETURN_REQUEST）**：START → COLLECT_ORDER_ID → VERIFY_ORDER → CHECK_ELIGIBILITY(deepseek-reasoner) → COLLECT_REASON → CONFIRM → EXECUTE → IReturnService.create_return() → 根据 result.status 回复（当前统一 APPROVED）。

**仅退款（REFUND_REQUEST）**：START → VERIFY_ORDER → CHECK_REFUND_ELIGIBILITY(deepseek-reasoner) → (PAID=可退 / SHIPPED=需先拒收 / DELIVERED=必须走退货) → EXECUTE

**投诉（COMPLAINT）**：START → COLLECT_COMPLAINT_TYPE → COLLECT_DESCRIPTION → SEVERITY_ASSESS(deepseek-reasoner) → EXECUTE

**全局控制**：所有节点可被 CANCEL / BACK / CHANGE / HUMAN 打断。

### 2.4 意图分类 Prompt（含 CHITCHAT + 上下文注入）

意图类别含 CHITCHAT。**意图分类 prompt 注入当前状态机上下文**，防止业务流中的短词被误判：

```
[当前状态] 用户正在确认退货 ORD-001，等待确认。如果输入为肯定语义（确认、好的、行等），
应归类为 RETURN_REQUEST 而非 CHITCHAT。
```

示例含"你好"→CHITCHAT、"我要退货"→RETURN_REQUEST missing_slots=["order_id"]。

CHITCHAT 在业务流中正确识别（如用户真的开始闲聊"今天天气不错"）→ 触发意图切换 → 保存快照 → 闲聊回复，流程通畅。

### 2.5 模型路由

| 任务 | 模型 | 延迟 |
|------|------|------|
| 意图分类 / 槽位填充 / 响应生成 / 闲聊 / RAG 问答 | deepseek-chat | 200ms-1s |
| 退货资格判定 / 仅退款资格判定 / 投诉严重性评估 | deepseek-reasoner | 1-3s |

默认 chat，仅资格判定/严重性评估走 reasoner（3s 超时降级 chat）。闲聊跳过状态机/RAG/对接层。前 2 轮 LLM 自由回复（友善 + 引导业务），第 3 轮温和收束，第 4 轮用规则话术兜底。

### 2.6 DeepSeek Gateway（Key 池化 + 排队 + 背压）

**问题**：单 Key RPM 上限 ≈200，大并发下需要控制并发而非加连接数。

**KeyState**：两态（healthy | cooling），RPM 滑动窗口，asyncio.Lock 保护，后台 30s 清理过期时间戳。

**KeyPool**：选 healthy 中 RPM 最低的 Key，无 healthy → 返回 None。

**Gateway 流程**：
- 有 healthy Key → 执行；收到 429 → mark_rate_limited(retry_after) → 冷却；收到 5xx → 换 Key 重试（最多 2 次），全失败抛 LLMUnavailableError
- 无 healthy Key → 排队等 2s → 排到执行 / 超时→CapacityExceededError
- 全部 cooling → AllKeysDownError → 降级规则引擎

```python
@dataclass
class KeyState:
    api_key: str
    _timestamps: deque = field(default_factory=deque)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    rpm_limit: int = 200
    status: str = "healthy"     # healthy | cooling
    cooldown_until: float = 0.0

    async def is_available(self) -> bool:
        async with self._lock:
            self._drop_expired()
            if self.status == "cooling":
                if time.time() >= self.cooldown_until: self.status = "healthy"
                else: return False
            return len(self._timestamps) < self.rpm_limit * 0.9

    async def get_rpm(self) -> int:
        async with self._lock: self._drop_expired(); return len(self._timestamps)

    async def record_request(self):
        async with self._lock: self._timestamps.append(time.time())

    async def mark_rate_limited(self, retry_after: int):
        async with self._lock:
            self.status = "cooling"
            self.cooldown_until = time.time() + retry_after
            self._timestamps.clear()

    def _drop_expired(self):  # 调用方必须持锁
        cutoff = time.time() - 60
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
```

配置从 .env 读取：`DEEPSEEK_PER_KEY_RPM`、`DEEPSEEK_QUEUE_MAX_SIZE`、`DEEPSEEK_QUEUE_TIMEOUT`。

### 2.7 规则引擎（LLM 全部不可用）

仅当 `KeyPool.all_cooling() == True` 启用，10 条正则规则 O(n) 匹配，零外部依赖，回复含客服热线。

---

## 3. 前端架构

- Vue 3 + TypeScript + Vite + Element Plus + Pinia
- HTTP POST 发消息，SSE 流式接收（status/token/action/done 四种事件）
- 仅在确认节点渲染 ConfirmButton，不做通用快捷回复
- Admin：同一套登录，users.role 区分权限，JWT payload 含 role → 前端根据 role 显隐菜单
- 开发模式：Vite dev server（localhost:5173）+ proxy /api → backend:8000；生产：nginx :80 统一入口，前端构建产物挂载 frontend/dist

---

## 4. 对接层架构

### 设计理念

Agent → ABC 接口 → Local 实现（操作 MySQL）→ 未来 Remote 实现（HTTP/gRPC 调微服务）。Agent 代码零改动。

### MySQL 表结构（完整，init.sql 中预置种子数据）

users(id, username, password_hash, role, phone, created_at)
orders(id, order_id, user_id, status, total_amount, shipping_address, created_at, delivered_at)
order_items(id, order_id, item_id, name, price, quantity, returnable,
           status ENUM('NORMAL','RETURN_REQUESTED','RETURNED','REFUNDED') DEFAULT 'NORMAL')
return_orders(id, return_id, order_id, user_id, items, reason, refund_amount, status DEFAULT 'APPROVED', session_id, created_at)
  -- UNIQUE(order_id, user_id) 防并发重复插入；写入用 INSERT IGNORE
refund_orders(id, refund_id, order_id, user_id, reason, amount, status DEFAULT 'APPROVED', session_id, created_at)
complaint_tickets(id, ticket_id, user_id, order_id, complaint_type, description, severity, status, session_id, created_at)
conversation_history(id, session_id, user_id, intent, messages, agent_state, summary, result, created_at)

种子数据（3 个用户 + 5 个订单 + 8 条商品明细，覆盖全场景）：

| 订单 | 用户 | 状态 | 金额 | 商品 | 距今 | 测试场景 |
|------|------|------|------|------|------|---------|
| ORD-20240801-001 | user_1 | DELIVERED | ¥69.70 | 手机壳+钢化膜 | 4天 | 正常退货 + 部分退货（2退1） |
| ORD-20240805-002 | user_1 | SHIPPED | ¥228.90 | 蓝牙耳机+耳机收纳盒 | 2天 | 多商品退货、仅退款被拒 |
| ORD-20240806-003 | user_1 | PAID | ¥89.85 | 数据线×2 + 定制手机支架(不可退) | 1天 | 仅退款成功 + returnable=false |
| ORD-20240720-004 | user_2 | CANCELLED | ¥150.00 | 充电宝 | 18天 | 订单已取消 |
| ORD-20240725-005 | user_1 | DELIVERED | ¥88.00 | 台灯 | 13天 | 超7天退货期 |

两个普通用户（user_1、user_2）各自归各自订单，测试用户隔离：user_2 不能退 user_1 的订单。

admin 账号密码 `123456`，bcrypt 预生成 hash 写入 init.sql。

### ABC 接口 + Local 实现（校验 order_items.status 防重复退）+ 写入重试 + 依赖注入

### 升级路径

```
当前: Agent → IOrderService → LocalOrderService → MySQL
未来: Agent → IOrderService → RemoteOrderService → HTTP/gRPC → 订单微服务
```

---

## 5. RAG 系统

- **知识库源数据（Phase 6 起）**：MySQL `knowledge_docs` 表为单一事实来源（source of truth），存原始 markdown 全文；ChromaDB 只存分块向量快照（派生索引），每次从源重新生成、不独立演进。启动时 MySQL 空 → 从 `backend/app/rag/knowledge/` 下 4 个 Markdown 灌入；ChromaDB 同步走"增量补偿 pending 行"，全量重建仅在向量库丢失/首次灌入时执行，避免全量对账成为常态路径

| 文件 | 内容 |
|------|------|
| `return_policy.md` | 7 天无理由退货、商品完好定义、不适用品类、运费规则 |
| `refund_policy.md` | 仅退款条件（未发货秒退 / 已发货拒收后退 / 已签收走退货）、退款时效 1-3 天 |
| `after_sales_policy.md` | 售后总则：审核时效 2h、质保维修、客服热线 400-XXX-XXXX |
| `faq.md` | 常见问题：退款到账时间、物流查询、退货包装要求、电子发票 |

内容编造合理即可。Admin 可查看/修改/删除/上传新文档（Phase 6 起为文档级管理，MySQL+ChromaDB 双同步）。

- **向量对接层**：IVectorStore 接口（add_documents / search / delete），当前 ChromaDB，未来可切 Milvus
- **ChromaDB**：独立容器服务（chromadb/chroma:1.4.4，backend 走 HttpClient 连接），数据持久化在 chroma_data volume，HNSW 索引 <50MB、检索 <10ms。
  开发单机可将 CHROMA_HOST 留空回退嵌入式 PersistentClient（config.chroma_host 切换）
- **缓存**：L1 Redis 精确缓存（TTL 600s），预期命中率 70%+
**空结果兜底**：检索 score < 0.3 或 0 条 → 回复"您的问题暂未收录，建议联系人工客服确认"，不注入空上下文到 prompt
**空集合检测**：启动时 `collection.count() == 0` → warn 日志 + Admin 页面显示提示"知识库为空，请上传政策文档"
- **Embedding**：bge-small-zh-v1.5，512 维，分块 512t + 128 overlap。query embedding 走 Redis 缓存；Admin 上传时离线 embedding。未来可切换 DeepSeek Embedding API（复用 Gateway Key 池），改一行配置

---

## 6. 会话管理

- session_id → Cookie（HttpOnly; Secure; SameSite=Strict），其余全部服务端
- Redis：主存储（session:{sid} + snapshot:{sid}:{intent}）
- MySQL：异步快照兜底（conversation_history）
- StorageRouter：Redis 不可用 → 自动切 MySQL → Redis 恢复后 5s 内切回
- **会话不存在**：带过期 session_id 的请求 → 自动创建新会话 → 回复"您好！请问有什么可以帮您？"
  （Phase 1 实现为返回 404，Phase 3.8 完善为自动创建新会话）
- **并发消息锁**：同一 session 并发请求 → asyncio.Lock(session_id) 串行化，防止状态覆盖
- **优雅关闭**：FastAPI shutdown 事件 → 停止接受新请求 → 等 30s → 保存活跃 session checkpoint → 关闭 DB/Redis/Gateway
- 上下文窗口：4000 tokens（System Prompt 500 + 摘要 200 + 最近对话 2000 + 状态 300 + 检索 1000）
- 安全：System Prompt 安全注入 + 预处理正则 → 两层 Prompt Injection 防护

---

## 7. 高并发策略

- **Nginx**：worker_connections 4096 × 4 = 16384 连接上限，高并发限流保护，超出 503
- **DeepSeek Gateway**：Key 池化 + RPM 追踪 + 排队 2s + 背压（详见 2.6）
- **缓存**：意图 60s / RAG 600s / 订单 30s / FAQ 3600s
- **容量**：FastAPI 8 workers、Redis 8GB+、MySQL 4C8G、10 Key × 200 RPM

---

## 8. 项目结构

```
customer-service/
├── docker-compose.yml / Makefile / .env.example / .gitignore
├── backend/app/
│   ├── api/ (routes, sse, auth, deps)
│   ├── agent/ (orchestrator, pipeline, intent, slots, context, response, rule_engine,
│   │           state_machine/, function_calling/, prompts/)
│   ├── rag/ (interfaces, embedder, chroma_impl, retriever, knowledge/)
│   ├── session/ (manager, storage_router, models)
│   ├── services/ (interfaces, local_impl, models)
│   ├── infrastructure/ (redis, mysql, deepseek_gateway, deepseek_keypool, circuit_breaker)
│   └── utils/ (logger.py - JSON 结构化)
├── frontend/src/
│   ├── views/ (Login, Register, CustomerService, Admin)
│   ├── components/ (chat/, order/, common/)
│   ├── composables/ (useChat, useSSE, useAuth, useSession)
│   └── api/ (client, authApi, sessionApi, adminApi)
└── docker/ (Dockerfile.backend, Dockerfile.frontend, nginx.conf)
```

---

## 9. API 设计

| Method | Path | 说明 |
|--------|------|------|
| POST | /api/v1/auth/register | 注册 |
| POST | /api/v1/auth/login | 登录，返回 JWT（含 role） |
| POST | /api/v1/sessions | 创建会话 |
| POST | /api/v1/sessions/{id}/messages | 发送消息 |
| GET | /api/v1/sessions/{id}/stream | **SSE 流式响应** |
| POST | /api/v1/admin/knowledge | 上传政策（admin） |
| POST | /api/v1/admin/orders | 录入订单+商品明细（admin） |
| PUT | /api/v1/admin/orders/{id} | 修改订单（admin） |
| DELETE | /api/v1/admin/orders/{id} | 删除订单（admin） |
| POST | /api/v1/admin/orders/{id}/items | 添加商品明细（admin） |
| DELETE | /api/v1/admin/orders/{id}/items/{item_id} | 删除商品明细（admin） |

SSE 协议：status → token* → done | error

---

## 10. 实施顺序

1. **基础设施 + 认证 + 简单对话**：Docker Compose、JSON 日志、JWT 注册登录、DeepSeek 客户端、Vue 脚手架
2. **意图分类 + RAG + Admin**：6 分类 + CHITCHAT、ChromaDB + 向量对接层 + 缓存、Admin 知识库上传
3. **状态机 + FC + 对接层**：ABC 接口 + Local 实现（含重试）、LangGraph ×3、FC 连接对接层、SSE 进度
4. **高并发优化**：DeepSeek Gateway（KeyPool+排队+背压）、Nginx 高并发限流、StorageRouter、规则引擎
5. **前端完善 + 测试**：SSE 进度条、ConfirmButton、pytest + vitest、README

---

## 11. 核心依赖

**后端**（按阶段追加，避免一次性拉大依赖）：
- Phase 1 已装：fastapi, uvicorn[standard], httpx, redis, asyncmy, sqlalchemy[asyncio], pydantic, pydantic-settings, bcrypt, PyJWT, python-multipart
- Phase 2 追加：chromadb, sentence-transformers
- Phase 3 追加：langgraph, langchain-core
- Phase 5 追加：pytest, pytest-asyncio

**前端**：vue 3, vue-router, pinia, axios, element-plus, typescript, vite, vitest

---

## 12. Docker Compose（nginx + backend×3 + redis + mysql, 无 Kafka）

```yaml
services:
  nginx:    (nginx:latest, :80, 前端静态 + /api/* 代理, proxy_buffering off, proxy_read_timeout 3600s)
  backend:  (FastAPI, Python 3.11-slim, 开发期 1 实例(Phase 4 扩 replicas:3), 连独立 chroma(HttpClient),
            env: SERVICE_MODE=local, DEEPSEEK_API_KEYS, CHROMA_HOST=chroma,
            logging: json-file, max-size:50m, max-file:3 — 防日志写满磁盘)
  chroma:   (chromadb/chroma:latest=1.4.4, 内部 :8000, PERSIST_DIRECTORY=/data, chroma_data volume,
            healthcheck: bash TCP 探测 8000)
  redis:    (redis:7, healthcheck)
  mysql:    (8.0, init.sql 自动建表+种子数据)
volumes: {mysql_data, chroma_data, redis_data, hf_cache}

备注（与设计差异）：
- Python 基础镜像用 3.11-slim（方案原写 3.12）：asyncmy 0.2.9 无 cp312 wheel，slim 镜像无编译链装不上
- 镜像标签为本机可用版本（nginx:latest/redis:7 替代 alpine）：本机 Docker Hub 拉取超时；有网络可换回 alpine
- ChromaDB 改为独立容器（1.4.4）：解决多实例共享 volume 并发写问题；后端客户端同步升级 chromadb 1.4.4 匹配协议
- chroma healthcheck 必须用 bash TCP 探测 8000（chroma 1.x 无 version 子命令，会导致 unhealthy）
- backend 开发期 1 实例；多实例扩展时 chroma 独立服务已支持
```

Nginx：`/api/` 全局高并发限流 + `/api/v1/auth/` 单独 5r/m（防暴力破解），`proxy_buffering off` 保证 SSE 逐帧推送。

---

## 13. 配置（.env）

```bash
DEEPSEEK_API_KEYS=sk-key1,sk-key2,sk-key3       # 多 Key 逗号分隔
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL_CHAT=deepseek-chat
DEEPSEEK_MODEL_REASONER=deepseek-reasoner
DEEPSEEK_PER_KEY_RPM=200                         # 单 Key RPM 上限
DEEPSEEK_QUEUE_MAX_SIZE=500                      # 排队容量
DEEPSEEK_QUEUE_TIMEOUT=2.0                       # 排队超时(s)
DEEPSEEK_TIMEOUT_CHAT=8.0 / DEEPSEEK_TIMEOUT_REASONER=15.0
SERVICE_MODE=local                               # local | remote
VECTOR_STORE=chroma                              # chroma | milvus
REDIS_URL=redis://redis:6379/0
MYSQL_URL=mysql+asyncmy://user:pass@mysql:3306/customer_service
MYSQL_POOL_SIZE=20 / (asyncmy 原生池无 max_overflow 概念，池上限即 pool_size，原 MYSQL_MAX_OVERFLOW 配置已弃用)
JWT_SECRET_KEY=change-me / JWT_EXPIRE_HOURS=2
SESSION_TTL=3600 / CONVERSATION_MAX_ROUNDS=10
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_HOST=chroma   # 空=嵌入式 PersistentClient；非空=HttpClient 连独立 chroma 服务
CHROMA_PORT=8000
RAG_CACHE_TTL=600 / INTENT_CACHE_TTL=60
ADMIN_DEFAULT_USERNAME=admin / ADMIN_DEFAULT_PASSWORD=123456
```

---

## 14. 日志规范

所有日志 JSON 结构化，统一 `logger.info/error` + `extra={}` 注入上下文。按系统分层：

### 请求级（每条 1 entry + 1 exit）

```json
{"event":"request_in","session_id":"sess_123","user_id":"u_456","input_len":42}
{"event":"request_out","session_id":"sess_123","intent":"RETURN_REQUEST","status":"ok","total_ms":1240,"tokens":85}
```

### Agent Pipeline（6 个阶段各 1 条）

| 阶段 | 字段 |
|------|------|
| 1.预处理 | `injection_detected`, `ms` |
| 2.意图 | `intent`, `confidence`, `intent_switched`, `ms` |
| 3.上下文 | `context_tokens`, `from_snapshot`, `ms` |
| 4.状态机 | `from_node→to_node`, `ms` |
| 5.动作 | `action_type(FC/RAG/LLM/none)`, `success`, `ms` |
| 6.响应 | `streaming`, `token_count`, `ms` |

### DeepSeek Gateway（每次 LLM 调用 1 条）

```json
{"event":"llm_call","model":"deepseek-chat","key_index":2,"key_rpm":45,"attempt":1,"ms":520,"status":"ok"}
```
特殊事件：`429`（标记冷却）、`5xx`（换 Key 重试）、`queue_wait`（排队时长）、`all_keys_down`（熔断触发）。

### 对接层（每次 DB 操作 1 条）

```json
{"event":"db_query","table":"orders","op":"SELECT","ms":3}
{"event":"db_write","table":"return_orders","return_id":"RC-xxx","ms":8,"retry":0}
```

### 会话管理

`session_created/loaded`、`storage_mode_switch`（Redis↔MySQL 切换）、`snapshot_saved/deleted:{intent}`

### RAG

`rag_cache_hit:L1` / `rag_cache_miss→chroma_ms+result_count+top_score` / `rag_empty` / `rag_collection_empty`

### Auth + Admin

登录成功/失败（含 username）、admin 操作（upload_knowledge、create_order、delete_order 含操作人）

### 系统健康（每 60s）

KeyPool：healthy_count / cooling_count / avg_rpm；按 intent 分组的 QPS

### 规则引擎

`rule_engine_triggered`：触发原因 + 匹配规则索引

---

## 15. 场景推演

### 场景 A：正常退货（完整闭环）✅
注册→登录→"我要退货 ORD-001"→追问原因→"质量问题"→确认→创建退单(APPROVED)→回复单号+退款时效。
**数据**：return_orders INSERT、order_items.status→RETURNED、snapshot 流程结束后删除。

### 场景 B：中途切换意图 ✅
退货到 COLLECT_REASON→"查 ORD-002 物流"→保存 snapshot:RETURN_REQUEST→切 ORDER_STATUS→完成→"继续退货"→恢复快照。
**数据**：snapshot 按 intent 存储/恢复，最大 6 个（每种意图一份）。

### 场景 C：CHITCHAT ✅
"你好"→LLM 自由回复→第 3 轮收束→第 4 轮规则话术。无 DB 写，无 reasoner 调用。

### 场景 D：RAG 命中 + 空结果 ✅
"退货条件"→命中→注入 prompt。"如何投诉物流"→score<0.3→"暂未收录，请联系人工"。
**数据**：命中时回写 L1 缓存；空结果不注入 prompt。

### 场景 E：DeepSeek 全部不可用→规则引擎 ✅
全部 Key cooling→熔断→规则引擎。"我要退货"→不命中具体规则→catch-all→"系统繁忙，请拨打 400-XXX-XXXX"。
**数据**：`rule_engine_triggered` 日志记录。

### 场景 F：并发消息（双击发送）✅
asyncio.Lock(session_id) 串行化→第二条读到第一条更新后的状态→不会重复创建。

### 场景 G：会话过期 ✅
session_id 过期→Redis 无→MySQL 无→返回 None→自动创建新会话→"您好！请问有什么可以帮您？"

### 场景 H：部分退货 ✅
退 SKU-001→status→RETURNED；SKU-002 保持 NORMAL→后续可单独退。return_orders UNIQUE(order_id,user_id) 防并发重复。

### 场景 I：重复退货被拒绝 ✅
SKU-001 已 RETURNED→check_eligibility→"该商品已退过"→NOT_ELIGIBLE→告知用户。

### 场景 J：仅退款资格判定 ✅
PAID(ORD-003) 可退 / SHIPPED(ORD-002) 需先拒收 / DELIVERED(ORD-001) 必须走退货。三级判定完整。

### 场景 K：订单已取消 ✅
查 ORD-004(CANCELLED)→"订单已取消，无法操作"。

### 场景 L：超 7 天退货期 ✅
查 ORD-005(DELIVERED 13天前)→check_eligibility→"已超过 7 天退货期，不可退"。

### 场景 M：用户隔离 ✅
user_2 退 user_1 的 ORD-001 → VERIFY_ORDER 校验 order.user_id ≠ 当前 user_id → "该订单不属于您的账号"。

### 场景 N：不可退商品 ✅
退 ORD-003 的定制手机支架(SKU-006, returnable=false)→check_eligibility→"该商品不支持退货"→NOT_ELIGIBLE。

---

## 16. 验证

```bash
docker-compose up -d
# 注册登录 → 创建会话 → SSE 流式对话 → 完整退货流程
pytest tests/ -v
# 意图准确率 >90%, 退货成功率 >95%, RAG 相关性 >80%, P99 <5s

---

## 17. 实施记录（Phase 1 已完成）

> 记录执行中的实际决策与验证结果，与设计文档的差异在此说明。

### 已实现并验证（2026-08-07）

- 服务栈 nginx/backend/redis/mysql 全部运行；`/healthz` OK
- MySQL 种子数据 3 用户 + 5 订单 + 8 商品明细（admin/123456、user_1/user_2 密码 123456）
- JWT 注册/登录（payload 含 role），admin 种子账号登录 OK
- Redis 会话 TTL 3600s 滑动过期（已验证）
- 简单对话 → DeepSeek 真实调用（4 个 key 已配置）中文回复 OK
- Nginx 反代 `/api/*`（:80）OK；Vite dev proxy `/api` → :8000 OK
- 前端 `vue-tsc` 类型检查通过；Vite dev server 运行于 :5173

### 实际决策与原因

| 项 | 方案原设计 | 实际 | 原因 |
|----|-----------|------|------|
| Python 基础镜像 | 3.12 | 3.11-slim | asyncmy 0.2.9 无 cp312 wheel |
| 镜像标签 | redis:7-alpine / nginx:alpine | redis:7 / nginx:latest | 本机 Docker Hub 拉取超时 |
| 种子用户 | 6 | 3（admin/user_1/user_2） | 订单只归属两个用户，多余无意义 |
| asyncmy 池 | pool_size / max_overflow | minsize / maxsize | asyncmy 继承 aiomysql 语义，无 overflow |
| 后端实例 | 3 replicas | 1（开发期） | ChromaDB 共享卷并发写需先解决 |
| 会话不存在 | 自动新建+问候语 | Phase 1 返回 404 | 兜底逻辑随 Phase 3.8 完善 |
| 路由前缀 | /api/v1/* | 认证 /api/v1/auth/*、业务 /api/v1/* | 避免 prefix 重复拼接 |

### Phase 2 补充：ChromaDB 改为独立容器（2026-08-07）

- 原因：嵌入式 PersistentClient 在多实例并发写 chroma_data volume 时有锁冲突，不符合"水平扩展"架构承诺
- 改动：compose 增加 chromadb/chroma:latest（1.4.4）独立服务；后端客户端升级 chromadb 0.5.23→1.4.4（版本匹配，0.5 客户端连不上 1.x 服务）；chroma_impl 增加 HttpClient 模式（config.chroma_host 切换）；chroma_data volume 数据因 1.x 存储格式不兼容而清空重建（重新 embedding 4 文档）
- 坑：chroma 1.x 无 `version` 子命令，healthcheck 用该命令会永远 unhealthy → backend 依赖卡死不启动 → 改用 bash TCP 探测 8000 端口
- 验证：chroma 服务 healthy；backend 连接成功并重建知识库（rag_init_done docs=4）；RAG 政策查询经独立服务返回正确答案

### Phase 3 补充：状态机 + FC + 对接层（2026-08-07）

- LangGraph 状态机执行模型：每轮 step 循环推进到"等待输入"节点（awaiting 标记），修正"每轮只推一个节点"的缺陷
- 关键坑：LangGraph state schema（TypedDict）必须声明所有节点返回字段（awaiting 缺失会被丢弃 → 状态机死循环）
- 意图切换 + 快照：切换时保存当前状态机，恢复时从 snapshot 续推

### Phase 4 补充：高并发 + 容错（2026-08-07）

- DeepSeek Gateway：Key 池化（RPM 滑动窗口）+ 排队背压 + 429 冷却 + 5xx 换 Key + 熔断降级规则引擎（并发 30 全成功、Key 分布均衡）
- StorageRouter：Redis 主存 + MySQL 双写，停 Redis 自动切 MySQL，恢复 5s 内切回（已实测）
- 并发锁：同一 session 串行化；优雅关闭：uvicorn 30s graceful + shutdown 日志
- 坑：pip 无 chromadb==1.4.4（1.4 系列止于 1.4.1）；Docker Windows build COPY 层缓存误判（需 --no-cache 强制）

### Phase 5 补充：测试 + 收尾（2026-08-07）

- 后端单元测试 23 通过（mock LLM/对接层/存储）；集成测试 3 通过（真实服务端到端退货流程）；前端组件测试 9 通过
- 14 场景走查：A-N 全覆盖，**验收发现并修复 2 个真实 bug**：
  1. 状态机 `_to_order` 丢 `delivered_at` → 超 7 天退货判定失效（场景 L）
  2. `_order_to_dict` 缺 `db_id` → 退货后商品状态更新失败 → 重复退货不拦截（场景 I）
- 压测：auth 端点 5r/m 限流保护生效（500 并发 493 被 503 拒）；业务端点 300 并发全成功；P99=27ms

### Phase 6 补充：Admin 知识库内容管理增强（2026-08-07）

背景：原知识库只有上传/删除（chunk 粒度），无法修改已上传内容。方案评审时确认：**要做文档级编辑，就必须有原始文档存储**——纯 ChromaDB 方案下 chunks 是"可变快照"，一旦改过就无法还原原文，数据完整性无解。用户建议"MySQL 存源文档 + ChromaDB 存向量"，采纳。

- **数据模型**：新增 `knowledge_docs(source UNIQUE, content, updated_by, sync_status, created_at, updated_at)`。MySQL 为源，ChromaDB 为派生索引，任何修改都从 MySQL 重新分块生成，ChromaDB 不独立演进
- **API**：`POST`(UPSERT 覆盖)/`GET`(文档级)/`PUT /{source}`(编辑整篇)/`DELETE /{source}`(文档级)/`POST /sync`(全量对账，含孤儿清理)
- **一致性策略**（避免全量对账压力）：写 ChromaDB 失败 → 该行 `sync_status='pending'` + 返回 502 → 每次写操作后**增量补偿**只重建 pending 行（O(失败数)）→ 全量对账仅 admin 手动触发或向量库丢失时启动重建，不进常态路径
- **前端**：知识库 tab 改文档级列表（同步状态 tag + 编辑弹窗 + "同步知识库"按钮）
- **验收发现并修复 2 个真实 bug**：
  1. `session_mysql_save_error: datetime not JSON serializable`——状态机把订单 `delivered_at`(datetime) 放进 agent_state 后，MySQL 会话兜底写入永远失败（被 try 吞掉，Redis 优先故无感）；`json.dumps(..., default=str)` 修复，新增单测回归
  2. ChromaDB 1.4 的 `count()` 不接受 where 参数（列表接口 500）→ 改用 `get(where=...)` 数 ids
- **验证**：后端单测 33 通过（含知识库管理 6 个 + datetime 序列化回归）；集成测试幂等化（跑完整退货流程前重置订单残留，支持重复跑）；API 集成验证 7 步全过；**一致性场景实测**（停 chroma → 上传 502 + MySQL pending → 恢复 → 下次写操作自动补偿为 ok）；Admin UI 9/9、用户端 UI 9/9 通过；前端 vue-tsc + build 通过

### Phase 6.2 补充：Admin 一键重置测试数据（2026-08-07）

背景：测试痛点——退货流程会"消费"种子订单（SKU→RETURNED、状态被改、新建订单），验证脚本重复测试需要手动清 DB。正规化为 Admin UI 按钮。

- **后端**：`POST /admin/reset-demo`（admin-only），事务内清空 return_orders/refund_orders/complaint_tickets/order_items/orders → 重插 init.sql 种子订单（5 订单 + 8 商品）。种子数据硬编码在 routes.py 并注释"与 init.sql 保持一致"
- **前端**：订单 tab 顶部"重置测试数据"按钮 + `ElMessageBox.confirm` 二次确认
- **设计取舍**：重置用"删光重插"而非"状态回退"——Admin 可能改状态/新建/删过种子订单，只有重插保证恢复初始。代价是 DB 自增 id 变化，但业务订单号 `ORD-xxx` 不变，聊天用订单号不受影响
- **验证**：API 集成验证（污染 2 SKU RETURNED + 1 退单 → reset → 全清 + 实际退货返回"请问退货原因"证明订单可用）；Admin UI 11/11（含重置两步）；80 入口发布形态全量复验通过

### 最终状态
- 5 阶段 34 任务 + Phase 6 增强全部完成；服务栈 nginx/backend/chroma/redis/mysql 全 healthy
- 唯一待用户人工项：浏览器体验完整流程（注册/登录/聊天/退货/管理后台）
```
