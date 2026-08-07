# AI 智能客服系统

高并发 AI Agent 智能客服系统：面向电商售后场景，支持**退换货政策查询、订单状态查询、退货、仅退款、投诉**等完整业务闭环，并具备从 **Agent 编排、RAG 检索、存储高可用到 LLM 网关治理**的全链路工程化能力。

**架构不妥协**：前端、后端、向量库、存储全部组件化，可独立水平扩展，为大并发峰值流量而设计。

---

## 目录

- [第一部分：定位与功能](#第一部分定位与功能)
- [第二部分：技术架构、核心流程与闪光点](#第二部分技术架构核心流程与闪光点)
- [第三部分：Docker 部署、运行与测试](#第三部分docker-部署运行与测试)

---

# 第一部分：定位与功能

## 1.1 项目定位

以真实电商售后的高频场景为蓝本，构建一套**生产级 AI Agent 客服系统**的完整形态：

- **业务深度**：不止于"问答"，而是用状态机驱动**可完成的业务动作**——退货、仅退款、投诉，每一步落到数据库，结果真实可查。
- **架构高度**：高并发不是口号——DeepSeek 多 Key 网关、RAG 缓存、存储双写容灾、熔断降级，每一层都为高并发做了针对性设计。
- **工程完整度**：JWT 认证、Admin 管理后台、SSE 流式交互、JSON 结构化日志、单元/集成/组件测试全覆盖。

## 1.2 核心业务功能

### 用户侧（对话式智能客服）

| 场景 | 触发示例 | 系统行为 |
|------|---------|---------|
| 退货 | 「我要退货 ORD-20240801-001」 | 校验订单归属 → 判定退货资格 → 收集原因 → 用户确认 → 创建退货单，商品状态同步流转 |
| 仅退款 | 「我想仅退款 ORD-20240806-003」 | 三级资格判定：未发货可退 / 已发货需先拒收 / 已签收必须走退货 |
| 投诉 | 「客服态度差我要投诉」 | 收集投诉类型与描述 → LLM 评估严重性 → 生成工单 |
| 订单查询 | 「查一下订单」 | 缺单号时自动列出最近订单；支持中途任意切换意图，进度快照可恢复 |
| 政策咨询 | 「退货时限是多久」 | RAG 检索知识库 → LLM 结合政策回答，命中/未命中都有兜底 |
| 闲聊 | 「你好」 | LLM 自由回复并引导业务；连续闲聊第 4 轮自动收束到规则话术 |

### 管理侧（Admin 后台，`role=admin`）

- **知识库管理**：文档级上传 / 编辑 / 删除 / 全量同步。MySQL 存原文、ChromaDB 存向量，一致性强。
- **订单管理**：订单 + 商品明细的增删改，可模拟任意订单状态用于测试。
- **一键重置测试数据**：清空业务流水，恢复种子订单，测试可无限重跑。

## 1.3 测试入口

- 普通用户：`user_1 / 123456`（预置 5 个订单覆盖全场景）
- 管理员：`admin / admin123`

推荐测试链路：**「我要退货 ORD-20240801-001」** → 回答原因 → 点确认 → 收到退货单号 → 中途切走再「继续退货」体验快照恢复 → 切 admin 体验知识库管理。

---

# 第二部分：技术架构、核心流程与闪光点

## 2.1 总体架构

```
用户 ←── HTTPS ──→ Nginx :80（高并发限流 + SSE 反代）
                     ├─ /        → 前端静态文件
                     └─ /api/*   → FastAPI Agent API（可多实例水平扩展）
                                    │
                    ┌───────────────┴────────────────┐
                    │   Agent Pipeline（6 阶段）       │
                    │  预处理→意图识别→上下文装配→状态推进 │
                    │  →动作执行→SSE 响应               │
                    │   ├─ DeepSeek Gateway（Key 池化+  │
                    │   │   RPM + 排队背压 + 熔断）      │
                    │   ├─ 对接层 ABC → Local(MySQL)    │
                    │   └─ RAG：ChromaDB + Redis 缓存   │
                    └───┬────────┬────────┬───────────┘
                        │        │        │
                   MySQL(业务    Redis(会话   ChromaDB(向量,
                    +知识库源)  主存+缓存)   派生索引)
```

核心数据流（以退货为例）：

```
「我要退货 ORD-001」
  → 意图识别 → RETURN_REQUEST
  → 状态机 VERIFY_ORDER → 查订单（MySQL）
  → CHECK_ELIGIBILITY（deepseek-reasoner 判定可退）
  → 收集原因 → 用户确认
  → EXECUTE → 创建退货单 → MySQL 写入
  → 回复：「退货单 RC-xxx 已创建，退款 ¥69.70 将在 1-3 个工作日内原路返回」
```

## 2.2 核心流程：6 阶段 Agent 流水线

```
INPUT → [1.预处理] → [2.意图识别+切换] → [3.上下文装配] → [4.状态推进] → [5.动作执行] → [6.响应生成] → SSE OUTPUT
```

| 阶段 | 职责 | 说明 |
|------|------|------|
| 1. 预处理 | Prompt 注入检测 | 正则零成本拦截，敏感词过滤，不消耗 LLM |
| 2. 意图识别 | 6 类意图分类 | deepseek-chat（~200ms），JSON 容错 + 2 次重试，意图切换判断 |
| 3. 上下文装配 | 加载会话 + 恢复快照 | Redis 加载，切换意图时恢复对应快照 |
| 4. 状态推进 | LangGraph 状态机 | 每轮推进到「等待输入」节点，全局可打断 |
| 5. 动作执行 | FC / 对接层 / RAG | 同步调用，延迟 5-10ms，闲聊跳过 |
| 6. 响应生成 | SSE 流式 | 同时推送阶段进度事件，前端展示进度条 |

**关键设计**：
- **意图切换 + 快照**：业务流中切到别的意图，自动保存当前进度（按意图各存一份），回复「已保存退货进度，先帮您查 ORD-002 的物流」；回来输入「继续退货」即可恢复。
- **防止误判**：状态机上下文注入意图分类 prompt，使「确认」「好的」等短词归类为业务意图而非 CHITCHAT。
- **槽位追问**：缺 order_id 等必填槽位时不进状态机，LLM 动态生成追问并附带最近订单列表。

## 2.3 业务状态机（LangGraph）

三个业务流，每个节点可被 `取消 / 返回 / 更换 / 转人工` 全局打断：

- **退货**：收集订单 → 验证归属 → 资格判定(reasoner) → 收集原因 → 确认 → 执行 → 通知
- **仅退款**：验证订单 → 三级资格判定(reasoner) → 执行
- **投诉**：收集类型 → 收集描述 → 严重性评估(reasoner) → 生成工单

**模型路由**：意图分类/响应/闲聊走 `deepseek-chat`（200ms-1s）；退货/退款资格判定、投诉严重性评估走 `deepseek-reasoner`（1-3s），超时降级 chat。

### 状态数据结构（LangGraph State）

状态机的核心是一个**可 JSON 序列化的 dict（TypedDict）**，在节点间流转，由每轮用户输入驱动推进。

**通用字段**（三个流程共用）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_id` | int | 会话用户（JWT 鉴权后注入） |
| `session_id` | str | 会话 ID |
| `order_id` | str \| 空 | 业务订单号（如 ORD-...，流程中收集） |
| `user_input` | str | 本轮用户输入（每轮注入） |
| `stage` | str | **当前节点名**，图的唯一路由依据 |
| `awaiting` | str \| None | 等待输入的节点名；非空表示正等待用户输入 |
| `message` | str | 回复文案（追问 / 确认 / 最终结果） |
| `final` | bool | 是否终态（流程结束） |

**业务字段**（按节点写入）：

| 字段 | 类型 | 何时写入 | 说明 |
|------|------|---------|------|
| `order` | dict | verify_order 后 | 订单快照（含商品明细，见下方结构） |
| `eligibility` | dict | check_eligibility 后 | 退货/退款资格判定结果 |
| `result` | dict | execute 后 | 退单/退款执行结果 |
| `reason` | str | collect_reason 后 | 退货/退款原因 |
| `complaint_type` / `description` / `severity` | str | 投诉流程 | 投诉类型 / 描述 / 严重性（HIGH/MEDIUM/LOW） |

**嵌套结构**（可直接落库/序列化）：

```python
# order：订单快照（query_order 结果转 dict）
{
  "db_id": 47,                          # 内部主键（创建退单时更新商品状态用）
  "order_id": "ORD-20240801-001",
  "status": "DELIVERED",
  "total_amount": 69.70,
  "shipping_address": "上海市...",
  "delivered_at": "2026-08-03T15:00:00",   # 参与"超7天退货期"判定
  "items": [
    {"item_id": "SKU-001", "name": "手机壳", "price": 29.90,
     "quantity": 1, "returnable": true, "status": "NORMAL"}
  ]
}

# eligibility：资格判定结果（check_eligibility 返回）
{
  "eligible": true,
  "refund_amount": 69.70,
  "items": [{"item_id": "SKU-001", "name": "手机壳", "price": 29.90, "quantity": 1}]
}

# result：执行结果（create_return / create_refund 返回）
{
  "success": true,
  "status": "APPROVED",
  "return_id": "RC-1786097506282",
  "refund_amount": 69.70,
  "message": "ok"
}
```

**执行模型**（每轮推进一个节点）：

```
第 N 轮用户输入
  → 注入 user_input 进 state
  → START 按 state.stage 路由到目标节点
  → 节点执行，返回部分更新（LangGraph 自动浅合并，推进 stage）
  → 节点返回 awaiting（等待输入）或 final（终态）则本轮结束，等待下一轮
```

以退货流程为例，state 随对话逐节点变化：

| 用户输入 | 经过节点 | stage | awaiting | 关键变化 |
|---------|---------|-------|----------|---------|
| 「我要退货 ORD-...」 | collect_order_id → verify_order | verify_order | — | 提取 order_id |
| （自动） | check_eligibility | collect_reason | reason | 写入 order、eligibility |
| 「质量问题」 | collect_reason | confirm | confirm | 写入 reason |
| 「确认」 | confirm → execute → notify | END | — | 写入 result，final=True |

## 2.4 技术闪光点

### ① DeepSeek Gateway：多 Key 池化 + 排队背压 + 熔断（高并发核心）
单 Key 有 RPM 上限，撑不起海量并发，但**控制并发比加连接数更重要**：
- **KeyPool**：多 Key 滑动窗口 RPM 追踪，选负载最低的 healthy Key；429 → 自动冷却，5xx → 换 Key 重试（最多 2 次）。
- **排队背压**：无 healthy Key 时排队 2s，超时返回容量告警，不无限堆积。
- **熔断降级**：全部 Key 冷却 → 触发规则引擎（10 条正则，O(n) 匹配，回复含客服热线），**系统永不因 LLM 故障而崩溃**。

### ② StorageRouter：Redis 主存 + MySQL 双写自动切换（存储高可用）
- Redis 为主存储（会话/快照/缓存），MySQL 异步兜底双写。
- Redis 故障 → 自动切 MySQL 模式；后台 5s 探测 Redis，恢复后自动切回，日志记录 `storage_mode_switch`。
- MySQL 兜底读取时重建最近 10 条消息 + 摘要，保证 LLM 上下文完整。

### ③ RAG 一致性：MySQL 为源 + ChromaDB 派生索引（数据完整性）
- **根除行业通病**：纯向量库方案下 chunks 是「可变快照」，改过就无法还原原文；本项目改为 **MySQL 存原始 Markdown（source of truth），ChromaDB 只存分块向量快照**，每次从源重新生成，不独立演进。
- **一致性策略**：写向量失败 → 标记 `pending` → 下次写操作增量补偿（O(失败数)），全量对账仅异常恢复时 admin 手动触发。
- 检索链路：Embedding(bge-small-zh) → ChromaDB Top-10 → Re-rank Top-3 → Redis 精确缓存（TTL 600s）；空结果（score<0.3）不注入 prompt，回复引导人工。

### ④ 并发与资源治理
- **同一会话串行锁**：`asyncio.Lock(session_id)` 防止双击发送导致状态覆盖。
- **SSE 断连自取消**：检测 `request.is_disconnected()`，自动保存进度并取消生成，防资源泄露。
- **优雅关闭**：停止接新请求 → 30s 宽限 → 保存活跃会话 checkpoint → 关闭连接池。
- **MySQL 写入重试**：连接错误/死锁指数退避（0.1/0.2/0.4s）3 次，失败统一兜底话术。

### ⑤ 对接层抽象：ABC 接口 + Local/Remote 实现（可演进架构）
Agent 只依赖 `IOrderService` / `IReturnService` / `IRefundService` / `IComplaintService` 等抽象接口，当前 Local 实现直连 MySQL，未来切 Remote（HTTP/gRPC 微服务）**Agent 代码零改动**。

### ⑥ 安全与可观测性
- **双层 Prompt Injection 防护**：预处理正则（零成本）+ System Prompt 安全注入。
- **JWT 认证**：payload 含 role，Admin 接口权限隔离，前端按 role 显隐入口。
- **JSON 结构化日志**：每个请求 1 进 1 出 + 6 阶段各 1 条 + LLM 调用/DB 操作全埋点，注入 `session_id` / `intent` / `latency_ms`，支持全链路追踪。

### ⑦ 验证指标
单元测试 33 通过、集成测试端到端跑通 14 个业务场景（A-N）、压测 300 并发业务端点全部成功、P99 = 27ms。

---

# 第三部分：Docker 部署、运行与测试

## 3.1 前置条件

- Docker + Docker Compose（本项目使用 `docker compose` v2 语法）
- Node.js ≥ 18（前端构建/开发）
- 一个或多个 DeepSeek API Key（[平台申请](https://platform.deepseek.com)）

## 3.2 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，必填两项：
#   DEEPSEEK_API_KEYS=sk-key1,sk-key2,...   # 多 Key 逗号分隔，越多并发越高
#   JWT_SECRET_KEY=<随机长字符串>            # 生产必须替换，如 python -c "import secrets;print(secrets.token_urlsafe(48))"
```

关键配置项：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEYS` | — | 必填，多 Key 逗号分隔，KeyPool 自动负载均衡 |
| `JWT_SECRET_KEY` | change-me | 生产必须改 |
| `SERVICE_MODE` | local | `local` 直连 MySQL / `remote` 切微服务对接层 |
| `CHROMA_HOST` | chroma | 容器内服务名；留空则回退嵌入式向量库 |
| `REDIS_URL` / `MYSQL_URL` | 服务名 | 容器网络内使用服务名，本地开发改为 localhost |

## 3.3 启动全部服务

```bash
docker compose up -d
docker compose ps          # 等待全部 healthy
```

启动 5 个容器：`nginx` / `backend` / `chroma` / `redis` / `mysql`。MySQL 首次启动自动建表 + 灌入种子数据，ChromaDB 自动灌入 4 篇预置政策文档。

## 3.4 访问入口

| 入口 | 地址 | 用途 |
|------|------|------|
| 前端（生产） | http://localhost:80 | nginx 统一入口，含 API 反代 |
| 后端 API | http://localhost:8000/api/v1 | 开发调试（含 SSE） |
| 前端（开发热更新） | http://localhost:5173 | Vite dev server |
| **MySQL** | `localhost:3306`（csuser/cspass，库 `customer_service`） | 外部工具验证数据 |
| **Redis** | `localhost:6379` | 外部工具验证数据 |
| **ChromaDB** | http://localhost:8001 | 外部工具验证向量库 |

> 说明：MySQL/Redis/ChromaDB 默认不向宿主机暴露端口，仅 backend 通过 Docker 内网访问。若需用 Navicat/redis-cli 等工具直连验证数据，本机测试场景可在 `docker-compose.yml` 中为对应服务增加 `ports` 映射（本仓库已为验证便利开启，生产环境建议移除）。

## 3.5 前端构建与运行

```bash
# 开发模式（热更新，proxy /api → backend:8000）
cd frontend
npm install
npm run dev                # 打开 http://localhost:5173

# 生产模式（构建产物挂载到 nginx，通过 :80 访问）
npm run build
```

## 3.6 单元测试

后端单元测试（mock LLM/对接层/存储，不产生真实外部调用）。**推荐在 backend 容器内执行**——依赖与 `.env` 连接串（redis/mysql/chroma 服务名）已就绪，最省事：

```bash
docker compose exec backend python -m pytest tests/ -v
```

> 本地执行需先安装 `backend/requirements.txt` 并把 `.env` 中 `REDIS_URL`/`MYSQL_URL`/`CHROMA_HOST` 改为本机可达地址，否则 pytest 收集时初始化 RAG/连接会失败。

覆盖范围：意图分类（6 类准确率 >90%）、退货/退款/投诉状态机（7 节点 + 三级判定）、RAG（命中/空结果）、StorageRouter 切换、知识库文档管理（含 datetime 序列化回归）。

前端组件测试：

```bash
cd frontend && npx vitest run
```

覆盖范围：ChatPanel 消息渲染、StreamingMessage 逐字追加、ConfirmButton、登录/注册表单验证。

## 3.7 集成测试

端到端集成测试（`tests/test_integration.py`）走**真实服务链路**：创建会话 → 发消息 → SSE 流式接收 → 验证退货单号落库。**依赖运行中的服务**，通过 `GET /healthz` 探测，服务未启动时自动跳过（不会误报失败）：

```bash
docker compose up -d                              # 先确保服务在跑
docker compose exec backend python -m pytest tests/ -v
```

单条命令即可验证完整退货/退款/投诉流程；测试已幂等化（跑前重置订单残留），可重复执行。

## 3.8 常用运维命令

```bash
docker compose up -d          # 启动全部服务
docker compose down           # 停止（数据卷保留）
docker compose ps             # 查看状态
docker compose logs -f backend  # 跟踪后端日志（JSON 结构化）
docker compose exec backend python -m pytest tests/ -v   # 后端测试
```

也可以使用仓库自带 `Makefile`：`make up` / `make down` / `make ps` / `make logs` / `make test`（Windows 无 make 时用等价 docker compose 命令）。

## 3.9 常见问题

- **首次启动慢**：MySQL 建表灌种子、ChromaDB 下载 embedding 模型（bge-small-zh）都需要时间，等 `docker compose ps` 全 healthy 再访问。
- **没有 DeepSeek Key**：`DEEPSEEK_API_KEYS` 留空时系统启动可正常，但所有 LLM 调用会走规则引擎兜底（仅能回复预置话术）。
- **想重置测试数据**：admin 登录后，管理后台 → 订单 tab → 「重置测试数据」，一键恢复种子订单。
