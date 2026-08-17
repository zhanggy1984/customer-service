# 智能客服系统 — 任务拆分

> 基于技术方案文档拆分为 5 个大阶段。**全部任务已完成并自动验收（2026-08-07）**。

---

## Phase 1：基础设施 + 认证 + 简单对话（预计 2 天）

### ✅ 1.1 项目骨架搭建

**要做的事**：
- 创建 `customer-service/` 顶层目录结构
- 编写 `docker-compose.yml`（nginx + backend + redis + mysql，无 Kafka）
- 编写 `docker/nginx.conf`（`/` 前端静态文件，`/api/*` 反向代理 + 高并发限流 + `/api/v1/auth/` 5r/m 限流 + `proxy_buffering off`）
- 编写 `docker/Dockerfile.backend`（Python 3.12 + pip install）
- 编写 `docker/Dockerfile.frontend`（多阶段：Vite build → nginx alpine）
- 编写 `Makefile`（`up` / `down` / `logs` / `test`）
- 编写 `.env.example` + `.gitignore`

**验收**：`docker-compose up` 后 nginx:80、backend:8000、redis:6379、mysql:3306 均 healthy。

### ✅ 1.2 MySQL 初始化

**要做的事**：
- 编写 `backend/sql/init.sql`：
  - 建表：users、orders、order_items、return_orders、refund_orders、complaint_tickets、conversation_history
  - return_orders 加 `UNIQUE(order_id, user_id)` 约束
  - 种子数据：admin（密码 admin123 bcrypt 预生成 hash）、user_1、user_2 + 5 个订单 + 8 种商品明细
  - order_items 含 `status` 字段（NORMAL / RETURN_REQUESTED / RETURNED / REFUNDED）
- 编写 `backend/app/infrastructure/mysql.py`：asyncmy 连接池（pool_size=20, overflow=40，从 .env 读）

**验收**：`docker-compose up` 后 MySQL 自动建表，`SELECT * FROM orders` 返回 5 条种子数据。

### ✅ 1.3 配置管理 + 日志基建

**要做的事**：
- 编写 `backend/app/config.py`：pydantic-settings 从 `.env` 加载全部配置
- 编写 `backend/app/utils/logger.py`：JSON 结构化日志（`JsonFormatter`），注入 `session_id`/`user_id`/`intent`/`latency_ms`
- 日志输出到 stdout + `RotatingFileHandler`（Docker logging max-size:50m max-file:3）

**验收**：`logger.info("test", extra={"session_id":"s1"})` 输出合法 JSON。

### ✅ 1.4 JWT 认证（后端）

**要做的事**：
- 编写 `backend/app/api/auth.py`：`POST /register`（bcrypt 哈希存储）、`POST /login`（返回 JWT，payload 含 `sub`/`username`/`role`）
- 编写 `backend/app/api/deps.py`：`get_current_user` 依赖注入（解析 Bearer token）
- Admin 中间件：校验 `role == "admin"`

**验收**：curl 注册 → 登录拿到 token → 带 token 请求受保护端点 → 200；无 token → 401。

### ✅ 1.5 JWT 认证（前端）

**要做的事**：
- 编写 `frontend/src/views/LoginView.vue`：用户名 + 密码 + 登录按钮
- 编写 `frontend/src/views/RegisterView.vue`：用户名 + 密码 + 确认密码
- 编写 `frontend/src/stores/authStore.ts`：Pinia 存储 JWT + user 信息
- 编写 `frontend/src/api/client.ts`：Axios 实例，请求拦截器自动带 `Authorization: Bearer <token>`
- 编写 `frontend/src/api/authApi.ts`：封装 register/login
- Vue Router：登录页 → 成功后跳转 `/chat`；未登录访问 `/chat` → 重定向 `/login`

**验收**：前端页面注册 → 登录 → 跳转聊天页；刷新页面登录状态保持。

### ✅ 1.6 DeepSeek 客户端封装（基础版）

**要做的事**：
- 编写 `backend/app/infrastructure/deepseek.py`：单 Key HTTP 调用 `POST /v1/chat/completions`（httpx.AsyncClient）
- 简单封装：`async def chat(messages, model, timeout) -> dict`
- 不做 Key 轮转、不做排队（Phase 4 再做 DeepSeek Gateway）

**验收**：调 `chat([{"role":"user","content":"你好"}])` → 返回 DeepSeek 响应 JSON。

### ✅ 1.7 会话管理（基础版）

**要做的事**：
- 编写 `backend/app/session/models.py`：Pydantic Session 模型
- 编写 `backend/app/session/manager.py`：`create_session` / `get_session` / `update_session` / `close_session`，基于 Redis JSON 存储
- TTL 3600s，每次访问刷新 TTL
- `session_id` 通过 `Set-Cookie` 返回前端（HttpOnly; Secure; SameSite=Strict）
- 会话不存在 → 返回 None（上层自动创建新会话）

**验收**：创建会话 → Redis 有 `session:{sid}` key → TTL 3600 → 关闭后删除。

### ✅ 1.8 简单对话（单轮，无状态机）

**要做的事**：
- 编写 `backend/app/api/routes.py`：`POST /sessions`、`POST /sessions/{id}/messages`
- 编写 `backend/app/agent/orchestrator.py`（骨架版）：收到消息 → 调 DeepSeek → 返回文本
- 暂不做意图分类、不做状态机、不做 Function Calling

**验收**：curl 发"你好" → 返回 LLM 回复文本。

### ✅ 1.9 Vue 脚手架 + 基础聊天 UI

**要做的事**：
- `npm create vite@latest frontend -- --template vue-ts` + Element Plus
- 编写 `frontend/src/views/CustomerServiceView.vue`：左侧会话列表 + 右侧聊天面板布局
- 编写 `frontend/src/components/chat/ChatPanel.vue`：消息列表（区分 user/assistant 气泡）
- 编写 `frontend/src/components/chat/ChatInput.vue`：输入框 + 发送按钮
- 编写 `frontend/src/composables/useChat.ts`：`sendMessage` → HTTP POST → 收到回复 → 追加到消息列表
- 编写 `frontend/src/composables/useSession.ts`：创建/切换/关闭会话
- 编写 `frontend/src/composables/useAuth.ts`：登录状态 + 路由守卫

**验收**：前端登录 → 进入聊天页 → 发消息 → 看到 LLM 回复。

---

## Phase 2：意图分类 + RAG + Admin（预计 1.5 天）

### ✅ 2.1 意图分类器

**要做的事**：
- 编写 `backend/app/agent/intent.py`：
  - `classify_intent(user_input, current_state_context)` → 6 分类
  - `_extract_json()` 处理 markdown 包裹和尾随文本
  - JSON 解析失败 → 2 次重试 → 仍失败 → CHITCHAT 兜底
  - `confidence` 范围校验（0.0-1.0）、`intent` 合法性校验
- 编写 `backend/app/agent/prompts/intent.py`：意图分类 System Prompt（6 类 + 状态机上下文注入 + few-shot 示例含 CHITCHAT）
- 编写 `backend/app/agent/slots.py`：`REQUIRED_SLOTS` 定义 + `check_missing_slots()` 返回追问提示

**验收**："我要退货"→RETURN_REQUEST missing_slots=["order_id"]；"你好"→CHITCHAT；输入乱码→CHITCHAT 兜底。

### ✅ 2.2 RAG 基建：知识库 + ChromaDB + 向量对接层

**要做的事**：
- 编写 4 个预置 Markdown 知识文档（`backend/app/rag/knowledge/`）
- 编写 `backend/app/rag/interfaces.py`：`IVectorStore` ABC（add_documents / search / delete）
- 编写 `backend/app/rag/chroma_impl.py`：`ChromaVectorStore(IVectorStore)`，PersistentClient 磁盘持久化
- 编写 `backend/app/rag/__init__.py`：依赖注入，`VECTOR_STORE=chroma` 启用 ChromaDB
- `docker-compose.yml` 加 `chroma_data` volume 挂载 `./data/chroma`
- 编写 `backend/app/rag/embedder.py`：bge-small-zh-v1.5，首次运行自动下载模型
- 启动时判断 `collection.count() > 0` 跳过 init，避免重复 embedding

**验收**：启动后 ChromaDB 有数据 → `collection.count() > 0` → Admin 页面可查看知识库文档列表。

### ✅ 2.3 RAG 检索 + 缓存

**要做的事**：
- 编写 `backend/app/rag/retriever.py`：`search(query)` → embedding → ChromaDB Top-10 → Re-rank Top-3
- Redis 精确缓存：`rag_cache:{md5}` TTL 600s，命中直接返回
- 空结果兜底：score < 0.3 或 0 条 → 不注入 prompt

**验收**："退货条件"→命中政策文档→缓存写入；"如何投诉物流"→空结果→不注入 prompt。

### ✅ 2.4 Admin 知识库管理 API

**要做的事**：
- 编写 `backend/app/api/routes.py` 补充：
  - `POST /admin/knowledge` → 上传 MD → 分段(512t+128 overlap) → embedding → `add_documents`
  - `DELETE /admin/knowledge/{id}` → `delete(ids=[id])`
  - `GET /admin/knowledge` → 列出已索引文档（从 ChromaDB metadata 读取）
- Admin 中间件校验 `role == "admin"`

**验收**：Admin 上传新政策 MD → 用户立即搜到新政策。

### ✅ 2.5 Admin 前端页面

**要做的事**：
- 编写 `frontend/src/views/AdminView.vue`：
  - 知识库管理：文档列表 + 上传按钮 + 删除按钮
  - 种子数据管理：订单列表 + 新建/编辑/删除订单表单（含动态商品明细行）
- 编写 `frontend/src/api/adminApi.ts`：封装所有 admin 接口
- HeaderBar 根据 `role == "admin"` 显隐 Admin 菜单入口

**验收**：Admin 登录 → 看到 Admin 菜单 → 上传政策→ 列表刷新 → 删除政策。

---

## Phase 3：状态机 + Function Calling + 对接层（预计 3 天）

### ✅ 3.1 对接层：ABC 接口 + 领域模型

**要做的事**：
- 编写 `backend/app/services/models.py`：OrderInfo、OrderItem、ReturnResult、RefundResult、ComplaintResult、ManualTicket dataclass
- 编写 `backend/app/services/interfaces.py`：IOrderService、IReturnService、IRefundService、IComplaintService、ITicketService ABC

**验收**：抽象类定义完整，包含 docstring 说明每个方法的契约。

### ✅ 3.2 对接层：Local 实现（MySQL CRUD）

**要做的事**：
- 编写 `backend/app/services/local_impl.py`：
  - `LocalOrderService`：query_order（含 items）、list_user_orders（含 user_id 过滤）
  - `LocalReturnService`：create_return（INSERT return_orders status=APPROVED，UPDATE order_items status→RETURNED，用 INSERT IGNORE 防并发重复）+ check_eligibility（查 order_items.status 防重复退、查 returnable 字段）
  - `LocalRefundService`：create_refund + check_refund_eligibility（三级判定：PAID→可退/SHIPPED→需拒收/DELIVERED→必须走退货）
  - `LocalComplaintService`：create_complaint
  - `LocalTicketService`：create_manual_ticket
- 编写 `backend/app/services/__init__.py`：依赖注入（SERVICE_MODE=local/remote）

**验收**：调 `order_service.query_order("ORD-001")` 返回完整 OrderInfo；调 `create_return(ORD-001, SKU-001)` → MySQL 写入 return_orders + order_items.status 更新。

### ✅ 3.3 对接层：MySQL 写入重试

**要做的事**：
- `_retry_on_db_error(fn, max_retries=3)` 装饰器：ConnectionError/DeadlockError → 指数退避（0.1s/0.2s/0.4s）→ 仍失败 → 抛 `ServiceUnavailableException`
- Agent 层捕获 → 回复"系统出问题了，请稍后重试"

**验收**：手动模拟 MySQL 断开 → 3 次重试 → 异常抛出 → Agent 回复正确。

### ✅ 3.4 Function Calling：工具注册 + 执行器

**要做的事**：
- 编写 `backend/app/agent/function_calling/registry.py`：6 个 tool schema（query_order / list_user_orders / check_return_eligibility / create_return_order / create_refund_order / create_complaint / search_policy）
- 编写 `backend/app/agent/function_calling/executor.py`：`execute(tool_name, params)` → 路由到对应 handler
- 编写 `backend/app/agent/function_calling/tools/`：每个 tool 的 handler，**内部调用对接层接口**（`IOrderService.query_order()` 等）
- `create_return_order` handler：调 LocalReturnService.create_return → 返回 ReturnResult（含 `status` 字段） → Agent 根据 status 动态生成回复（APPROVED→告知单号+退款时效 / PENDING→告知审核中）
- `create_refund_order` handler：同理，根据 RefundResult.status 回复

**验收**：LLM 调 `query_order("ORD-001")` → executor 路由到 order_tools → LocalOrderService → 返回 JSON → 注入 LLM 上下文。

### ✅ 3.5 LangGraph 状态机：退货流程

**要做的事**：
- 编写 `backend/app/agent/state_machine/base.py`：LangGraph StateGraph 基础定义
- 编写 `backend/app/agent/state_machine/return_flow.py`：7 节点（START→COLLECT_ORDER_ID→VERIFY_ORDER→CHECK_ELIGIBILITY→COLLECT_REASON→CONFIRM→EXECUTE→NOTIFY）
  - CHECK_ELIGIBILITY 节点：调 deepseek-reasoner + IReturnService.check_eligibility
  - CONFIRM 节点：用户确认语义判断（"确认"/"好的"/"行"等）
  - EXECUTE 节点：调 IReturnService.create_return，返回 result.status → Agent 生成对应回复
  - 全局打断：CANCEL / BACK / CHANGE / HUMAN

**验收**：输入"我要退货 ORD-001"→完整走完 7 节点→收到"退货单 RC-xxx 已创建，退款 ¥29.90 将在1-3个工作日内原路返回"。

### ✅ 3.6 LangGraph 状态机：仅退款 + 投诉

**要做的事**：
- 编写 `backend/app/agent/state_machine/refund_flow.py`：CHECK_REFUND_ELIGIBILITY 节点（deepseek-reasoner + IRefundService.check_refund_eligibility 三级判定）
- 编写 `backend/app/agent/state_machine/complaint_flow.py`：SEVERITY_ASSESS 节点（deepseek-reasoner 严重性分级）+ complaint_type 由 LLM 从输入提取
- 编写 `backend/app/agent/state_machine/edges.py`：全局边逻辑

**验收**：ORD-003(PAID)→仅退款成功；ORD-002(SHIPPED)→仅退款被拒提示先拒收。

### ✅ 3.7 Agent Pipeline 串联（6 阶段）

**要做的事**：
- 编写 `backend/app/agent/pipeline.py`：Pipeline 抽象类，6 个阶段按序执行
- 编写 `backend/app/agent/orchestrator.py`（完整版）：
  - 阶段 1：注入检测 + 空消息过滤
  - 阶段 2：意图分类（含 JSON 校验+重试）+ 意图切换判断 + 状态机上下文注入防止"确认"误判 CHITCHAT
  - 阶段 3：上下文装配 + snapshot 恢复
  - 阶段 4：LangGraph 状态推进
  - 阶段 5：FC 执行 + RAG 检索 + LLM 推理
  - 阶段 6：SSE 流式响应（含进度事件）
- 编写 `backend/app/agent/context.py`：上下文装配器
- 编写 `backend/app/agent/response.py`：SSE 响应生成器，检测 `request.is_disconnected()`

**验收**：请求完整走过 6 个阶段，SSE 流式输出。

### ✅ 3.8 Agent 兜底：会话不存在 + 订单列表为空

**要做的事**：
- 会话不存在 → 自动创建新会话 + 问候语
- `list_user_orders` 返回空列表 → "您最近没有订单，请直接告诉我订单号"
- CHITCHAT 连续 4 轮 → 规则话术兜底

**验收**：过期 session_id → 自动创建新会话；空订单列表 → 友好提示。

### ✅ 3.9 前端 SSE 流式对话 + ProgressBar + ConfirmButton

**要做的事**：
- 编写 `frontend/src/composables/useSSE.ts`：`EventSource` 连接 `/api/v1/sessions/{id}/stream`，处理 status/token/done/error 四种事件
- ChatPanel 渲染 ProgressBar（收到 status 事件时显示"正在查询订单..."，done 时消失）
- StreamingMessage 逐字渲染 token
- ConfirmButton：仅在 CONFIRM_RETURN / CONFIRM_REFUND / CONFIRM_COMPLAINT 节点 SSE 发送 action 事件时渲染
- OrderCard：SSE action 事件驱动渲染订单详情卡片

**验收**：发"我要退货 ORD-001"→看到进度条（"正在查询订单..."→"正在核实资格..."）→确认按钮→点击→收到退单号。

---

## Phase 4：高并发 + 容错（预计 1.5 天）

### ✅ 4.1 DeepSeek Gateway（Key 池化 + 排队 + 背压）

**要做的事**：
- 编写 `backend/app/infrastructure/deepseek_keypool.py`：KeyState（RPM 滑动窗口 + asyncio.Lock + 两态 healthy/cooling）+ KeyPool（select_key 选 RPM 最低的 healthy Key + 后台 30s 清理）
- 编写 `backend/app/infrastructure/deepseek_gateway.py`：DeepSeekGateway（有 healthy→执行 / 无 healthy→排队 2s / 全部 cooling→AllKeysDownError）+ 收到 429→mark_rate_limited + 收到 5xx→换 Key 重试最多 2 次
- 替换 Phase 1 的简单 DeepSeek 客户端

**验收**：配置 3 个 Key → 并发 LLM 调用 → 观察到 RPM 均衡分布 → 模拟 429 → Key 进入 cooling。

### ✅ 4.2 熔断器 + 规则引擎

**要做的事**：
- 编写 `backend/app/infrastructure/circuit_breaker.py`：`should_fallback(pool)` → `KeyPool.all_cooling() == True`
- 编写 `backend/app/agent/rule_engine.py`：10 条正则规则，O(n) 匹配，回复含客服热线
- Agent Pipeline：LLM 不可用时走规则引擎

**验收**：手动全部 Key cooling → 熔断触发 → "我要退货"→规则引擎回复（不崩）。

### ✅ 4.3 StorageRouter（Redis/MySQL 自动切换）

**要做的事**：
- 编写 `backend/app/session/storage_router.py`：
  - 双写（Redis 同步 + MySQL 异步）
  - Redis 不可用 → mode=mysql_fallback → 读写 MySQL
  - 后台 5s ping Redis → 恢复后自动切回
  - MySQL 读取时重建：messages[-10:] + summary → 完整 LLM 上下文

**验收**：停止 Redis→会话自动走 MySQL→启动 Redis→5s 内自动切回。日志记录 `storage_mode_switch`。

### ✅ 4.4 优雅关闭

**要做的事**：
- `app/main.py` 注册 `@app.on_event("shutdown")`：
  - 停止接受新请求
  - 等待活跃请求完成（graceful_timeout 30s）
  - 保存所有活跃 session checkpoint 到 MySQL
  - 关闭 DB/Redis/Gateway 连接池

**验收**：`docker-compose stop backend` → 日志显示"shutting down"→ 30s 内正常退出。

### ✅ 4.5 并发锁

**要做的事**：
- `asyncio.Lock` 字典，key=session_id → 同一 session 的并发请求串行化

**验收**：快速双击发送 → 第二条消息等待第一条完成 → 不出现状态覆盖。

---

## Phase 5：测试 + 文档 + 收尾（预计 1 天）

### ✅ 5.1 后端单元测试

**要做的事**：
- `test_intent.py`：6 分类准确率 >90%（准备 30+ 条测试用例，覆盖正常/边界/异常输入）
- `test_state_machine/test_return_flow.py`：7 节点 + 4 个全局边覆盖
- `test_state_machine/test_refund_flow.py`：三级判定覆盖
- `test_state_machine/test_complaint_flow.py`：严重性分级覆盖
- `test_rag.py`：命中/空结果 score<0.3 覆盖
- `test_session.py`：StorageRouter 切换覆盖

**验收**：`pytest tests/ -v` 全部通过。

### ✅ 5.2 后端集成测试

**要做的事**：
- `test_integration.py`：端到端退货流程（创建会话 → 发消息 → SSE 流式接收 → 验证退单号）
- `conftest.py`：Docker 容器 fixture

**验收**：单条命令跑通完整退货/退款/投诉流程。

### ✅ 5.3 前端组件测试

**要做的事**：
- vitest + @vue/test-utils
- 覆盖：ChatPanel 消息渲染、StreamingMessage 逐字追加、ConfirmButton 点击发送"确认"
- LoginView/RegisterView 表单验证

**验收**：`npx vitest run` 全部通过。

### ✅ 5.4 README + API 文档

**要做的事**：
- README.md：项目简介 + 技术栈 + `docker-compose up` 快速启动 + 架构图 + 测试步骤
- API 文档：9 个端点（含 curl 示例）
- .env.example：注释完整的所有配置项

**验收**：新开发者按 README 操作能在 5 分钟内跑起来。

### ✅ 5.5 最终验收

**要做的事**：
- `docker-compose up` 一键启动
- 脚本走通全部 14 个场景（A-N）
- 性能：locust 500 并发压测，P99 < 5s

**验收**：README 中的测试步骤全部跑通；压测报告截图。

---

## Phase 6：Admin 知识库内容管理增强（2026-08-07）

### ✅ 6.1 知识库源数据入 MySQL（MySQL 为源，ChromaDB 为派生索引）

**要做的事**：
- 新增表 `knowledge_docs(source UNIQUE, content, updated_by, sync_status, created_at, updated_at)`，存原始 markdown 全文
- ChromaDB 只存分块向量快照，永远从 MySQL content 重新生成，不独立演进（根除"chunk 拼接还原原文"的数据完整性缺陷）
- 启动初始化改为：MySQL 空 → 从 `knowledge/*.md` 灌入；ChromaDB 同步走增量补偿，全量重建仅向量库丢失时执行

**验收**：启动后 MySQL 有 4 篇文档，列表接口返回文档级数据（source/chunk_count/sync_status）。

### ✅ 6.2 文档级管理 API（上传/编辑/删除/全量对账）

**要做的事**：
- `POST /admin/knowledge`：UPSERT 语义（title 已存在则覆盖）
- `PUT /admin/knowledge/{source}`：编辑整篇内容
- `DELETE /admin/knowledge/{source}`：文档级删除（MySQL 权威，ChromaDB 尽力清理，孤儿由对账兜底）
- `POST /admin/knowledge/sync`：全量对账（重建全部 + 孤儿清理），admin 手动触发

**验收**：上传/编辑/删除/同步全链路 curl 验证通过，列表恢复 4 篇。

### ✅ 6.3 一致性策略：pending 标记 + 增量补偿 + 手动对账

**要做的事**：
- 写 ChromaDB 失败 → 该行 `sync_status='pending'` → 返回 502（提示点击同步按钮）
- 每次知识库写操作后增量补偿 pending 行（只重建失败的文档，O(失败数)，不做全量扫描）
- 全量对账不进常态路径，仅异常恢复时由 admin 手动触发

**验收**：停 chroma 容器 → 上传 502 + MySQL pending → 恢复 → 下次写操作自动补偿为 ok。

### ✅ 6.4 Admin 前端文档级管理

**要做的事**：
- 知识库 tab 改为文档级列表（标题/块数/同步状态 tag/更新时间/操作人）
- 编辑弹窗（标题只读 + 内容编辑 + 保存）
- "同步知识库"按钮（触发全量对账，显示重建/清理结果）

**验收**：Admin UI 验证 9/9 通过（文档列表/同步按钮/编辑弹窗/订单中文状态）。

### ✅ 6.5 附带修复

**要做的事**：
- `session_mysql_save_error: datetime not JSON serializable`：agent_state 含订单 delivered_at(datetime) 导致 MySQL 会话兜底写入静默失败；`json.dumps(..., default=str)` 修复
- ChromaDB 1.4 的 `count()` 不支持 where → 改用 `get(where=...)` 数 ids

**验收**：后端单测 33 通过（含新增 datetime 序列化 + 知识库管理 6 个测试）；集成测试幂等化（跑完整退货流程前重置订单残留）。

### ✅ 6.6 Admin 一键重置测试数据

**要做的事**：
- `POST /admin/reset-demo`：事务内清空退货/退款/工单/商品/订单，重插 init.sql 种子订单（5 订单 + 8 商品），admin-only
- Admin 订单 tab 顶部"重置测试数据"按钮 + `ElMessageBox.confirm` 二次确认
- 解决测试痛点：退货流程会"消费"种子订单（SKU→RETURNED），一键回到可测试状态

**验收**：污染状态（2 SKU RETURNED + 1 退单）→ reset → 全清 + 订单恢复可退（实际退货返回"请问退货原因"）；Admin UI 11/11 通过。
