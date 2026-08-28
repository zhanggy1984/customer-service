# AI 智能客服系统

> **高并发 AI Agent 智能客服系统**：面向电商售后场景，从**退换货政策问答、订单状态查询**，到**退货、仅退款、投诉**等可完成的业务动作，在对话内一站式闭环，并具备从 **Agent 编排、LangGraph 状态机、RAG 检索、存储高可用到 LLM 网关治理**的全链路工程化能力。

三句话理解这个系统：

- **做什么**：把电商售后搬进对话。用户在聊天里直接办理退货、仅退款、投诉，也能查订单、问政策——每一步真实落库，退单号/退款/工单号可查，不再是"只能聊不能办"的问答机器人。
- **怎么做**：LangGraph 状态机驱动三个可完成的业务流（退货/退款/投诉），DeepSeek 多 Key 网关 + RAG 检索生成答复，LLM 自主决策调只读工具（P3）+ 实时护栏（P4）+ 判定落库（P5）；SSE 契约化流式全链路可观测，LLM/存储故障一律熔断降级到规则引擎，**系统永不因 AI 故障崩溃**。
- **好在哪**：政策有依据不编造、业务动作对话内闭环、故障不崩；对外是契约对齐的 SSE 事件流，有 386 项后端测试（含真实链路集成）+ 41 项前端测试 + E2E 脚本，可直接复验。

## 目录

- [一、这是什么](#一这是什么)
- [二、系统架构](#二系统架构)
- [三、快速开始](#三快速开始)
- [四、使用场景与示例](#四使用场景与示例)
- [五、技术闪光点](#五技术闪光点)
- [六、技术栈一览](#六技术栈一览)
- [七、配置说明](#七配置说明)
- [八、目录结构](#八目录结构)
- [九、测试与验收](#九测试与验收)
- [十、开发指南](#十开发指南)
- [十一、常见问题](#十一常见问题)
- [十二、已知限制与优化方向](#十二已知限制与优化方向)
- [十三、版本记录](#十三版本记录)
- [附录：状态机数据契约（LangGraph State）](#附录状态机数据契约langgraph-state)

---

## 一、这是什么

电商售后的高频咨询与业务办理长期依赖人工客服，存在四个典型痛点：

- **人力成本高**：退货/退款/投诉等重复性售后占客服工单大头，逐单人工处理，成本随订单量线性增长；
- **政策口径不统一**：退货时限、仅退款资格、定制商品规则散落在客服记忆与 Excel 里，同一问题不同人答案不一，易引发客诉；
- **业务动作断层**：传统问答机器人能"说"不能"做"，退货/退款仍需跳转人工工单，对话内无法闭环；
- **峰值流量易崩**：LLM 单 Key 限流、Redis/MySQL 单点故障、向量库结果不可信，大促流量下系统直接不可用。

本系统针对以上四个痛点，提供四大核心能力：

| 能力 | 实现 | 对应痛点 |
|------|------|---------|
| **对话式业务办理** | LangGraph 状态机驱动退货/仅退款/投诉，每步落库真实可查，支持部分退货多轮指定 | 人力成本 |
| **政策问答** | MySQL 原文源 + Milvus 向量 + RAG 检索，命中/未命中都有兜底，绝不编造 | 口径不一 |
| **订单/进度查询** | 对接层抽象 + 意图切换快照，对话内随时查单、中途切流、断点恢复 | 业务断层 |
| **高可用工程** | DeepSeek 多 Key 网关 + StorageRouter 双写 + 分布式锁/熔断信号 Redis 外置（多节点自由扩缩容），LLM/存储故障系统不崩 | 峰值崩溃 |

> **一句话理解 Agent 编排**：系统把"看懂用户要什么"（意图识别）、"要不要查/调工具"（LLM 工具决策循环 + 护栏）、"怎么一步步办成业务"（LangGraph 状态机）、"回答不上来怎么办"（规则引擎兜底）四件事分层交给不同组件——LLM 负责理解与生成，规则负责确定性与兜底，状态机负责业务闭环。

---

## 二、系统架构

```mermaid
graph TB
    WEB["Vue3 + Element Plus 前端（frontend）"]
    NGINX["nginx :80<br/>静态服务 + /api/v1 反代 + TLS"]
    GATEWAY["统一 API 网关 api-gateway:8099（共享 infra）<br/>Host 虚拟域名路由 + X-Request-ID traceId<br/>按真实 IP 限流 + SSE 透传"]

    subgraph 交互层[交互层 Interaction]
        API["FastAPI Agent :8000<br/>REST API + SSE 流式<br/>认证鉴权 / 参数校验 / 结果格式化"]
    end
    subgraph 控制层[控制层 Control]
        ORCH["orchestrator 6 阶段流水线<br/>预处理→意图识别→上下文装配→状态推进→动作执行→SSE 响应"]
        INTENT["意图分类<br/>deepseek-chat + 规则前置短路"]
        SM["LangGraph 状态机<br/>退货 / 仅退款 / 投诉（业务动作闭环）"]
        LOOP["LLM 工具决策循环 + 实时护栏<br/>只读工具自主决策（allow/reject/override）"]
        RULE["规则引擎兜底<br/>LLM 熔断时 10 条正则接管"]
    end
    subgraph 能力层[能力层 Capability]
        SVC["services/interfaces（ABC）<br/>IOrderService / IReturnService / IRefundService / IComplaintService"]
        IMPL["Local 实现直连资源层<br/>可切 Remote 微服务，Agent 零改动"]
    end
    subgraph 资源层[资源层 Resource]
        FACADE["infrastructure 门面<br/>llm_gateway / mysql_pool / retriever / cooldown"]
        GW["DeepSeek Gateway<br/>多 Key 池化 + RPM + 排队背压 + 熔断"]
        RAG["RAG 检索<br/>bge-small-zh Embedding + Milvus Top-10 → Re-rank Top-3 → 缓存"]
        SESS["会话存储<br/>Redis 主存 + MySQL 兜底 + 分布式锁 + 快照"]
        MYSQL[(MySQL 8<br/>业务事实 + 知识库原文源)]
        REDIS[(Redis<br/>会话主存 + 缓存 + 锁信号)]
        MILVUS[(Milvus<br/>知识向量派生索引)]
    end

    WEB --> NGINX --> GATEWAY
    GATEWAY -- SSE 流式回复 --> WEB
    GATEWAY --> API
    API --> ORCH
    ORCH --> INTENT
    ORCH --> SM
    ORCH --> LOOP
    ORCH --> RULE
    ORCH --> SVC
    SVC --> IMPL
    IMPL --> FACADE
    ORCH --> FACADE
    FACADE --> GW
    FACADE --> RAG
    FACADE --> SESS
    GW --> MYSQL
    RAG --> MILVUS
    SESS --> REDIS
    SESS --> MYSQL
```

**分层架构（四层，单向依赖）**：后端按 **交互 → 控制 → 能力 → 资源** 四层组织——上层依赖下层、下层绝不反向依赖；层间经抽象接口交互，控制权在控制层（能力层与资源层被动响应调用），数据流自上而下单向流动：

| 层 | 职责 | 对齐代码 |
|----|------|---------|
| **交互层** | 接收请求、认证鉴权、参数校验、结果格式化 | `app/api/`（REST 路由 / auth / contracts） |
| **控制层** | 意图理解、对话状态、任务编排、路由到能力层 | `app/agent/`（orchestrator / intent / 状态机 / 决策循环+护栏 / 规则兜底） |
| **能力层** | 接收控制层指令执行业务操作、返回结果（被动响应） | `app/services/`（interfaces ABC + local 实现） |
| **资源层** | 工具 / 数据 / 记忆的抽象接口 | `app/infrastructure/`（门面）+ `app/rag/` + `app/session/` |

四个约束在代码中的落地：

- **单向依赖**：`api → agent → services → infrastructure`，资源层不反向 import 上层；
- **依赖抽象**：控制层只经 `infrastructure` 门面访问 `llm_gateway / mysql_pool / retriever`，能力层只依赖 `services/interfaces` 抽象（ABC），不直接引用具体实现；
- **控制反转**：LangGraph 状态机 + orchestrator 持有控制权，能力/资源组件不主动发起请求，只响应调用；
- **数据流单向**：`AgentState` 自上而下流转，下层仅返回结果供上层决策，无旁路回调。

**对外链路（统一 API 网关）**：浏览器只访问前端 nginx；nginx 将 `/api/v1` 反代到共享网关 `api-gateway:8099`（`Host: cs.local`），网关按 Host 虚拟域名路由到本 agent 后端，并生成 `X-Request-ID`（后端日志 `trace_id` 即此值）、按真实 IP 限流、SSE 透传。网关由共享 infra 仓库提供（`infra/api-gateway/`），未知 Host 一律 403 防串线。宿主端口映射的 backend 地址（如 `localhost:8000`）仅供开发调试直连，绕过网关。

**核心链路（以退货为例）**：

```
「我要退货 ORD-20240801-001」
  → 预处理（Prompt 注入检测，零成本正则）
  → 意图识别 → RETURN_REQUEST（deepseek-chat，~200ms，JSON 容错 + 2 次重试）
  → 上下文装配（Redis 加载会话 + 恢复快照）
  → 状态推进（LangGraph：collect_order_id → verify_order → check_eligibility）
  → 动作执行（对接层查订单 / 确定性资格判定 / chat 严重性评估 / RAG 检索）
  → SSE 响应（tool_call → reasoning → token 逐段流式 → usage → done）
  → 确认后 EXECUTE 创建退货单落库
  → 回复：「退货单 RC-xxx 已创建，退款 ¥69.70 将在 1-3 个工作日内原路返回」
```

**编排模式：LLM 自主决策 + 规则护栏（P3-P5）**：对订单查询/政策问答两类意图，LLM 带只读工具列表自主决定调哪些工具（`search_policy`/`query_order`/`list_user_orders`），护栏在决策与执行之间做确定性校验（`allow/reject/override` 三态）；业务副作用工具（建退货/退款/投诉单、资格判定）的决策一律被护栏拦截，转由 LangGraph 状态机确定性接手——**决策与执行分离，LLM 不裸调业务动作**。每轮护栏判定落库 `tool_call_log`，为管理侧调用分析预留数据底座。

---

## 三、快速开始

> ⚠️ **前置依赖：共享 infra**。本 agent **不自带任何中间件**（MySQL/Redis/Milvus/MinIO 等），运行前须先部署共享 infra 仓库：
>
> ```bash
> # 发布物：clone infra 独立仓库后启动
> git clone https://github.com/zhanggy1984/share-infra && cd infra && docker compose up -d
> # 本地开发：infra 位于 ../infra
> cd ../infra && docker compose up -d
> ```

前置：Docker + Docker Compose（`docker compose` v2 语法）、Node.js ≥ 18。

### 第 1 步：配置环境变量

```bash
cp .env.example .env
# 编辑 .env，必填三项：
#   DEEPSEEK_API_KEYS=sk-key1,sk-key2,...   # 多 Key 逗号分隔，越多并发越高
#   JWT_SECRET_KEY=<随机长字符串>            # 生产必须替换，如 python -c "import secrets;print(secrets.token_urlsafe(48))"
#   ADMIN_DEFAULT_PASSWORD=<强密码>          # admin 密码（弱口令/留空会启动 fail-fast，见 §7）
```

**HTTPS 证书说明**（避免误判：自签证书 ≠ 系统不可上线）：
- 仓库自带**自签证书**（`certs/`，一年有效，SAN 含 `localhost`/`127.0.0.1`）——**仅供本地/内网环境使用**。浏览器首次访问会提示不受信任，这是**浏览器对自签证书的正常告警**，不是系统故障，点「高级 → 继续访问」即可；
- 内网 IP / 自定义域名访问：`bash scripts/gen_cert.sh 192.168.1.10`（把访问域名加进 SAN，避免 `NET::ERR_CERT_COMMON_NAME_INVALID`）；
- **公网上线必须换受信任证书**（约 5 分钟）：Let's Encrypt 颁发后，把 `cert.pem`/`privkey.pem` 复制为 `certs/server.crt`/`server.key`（路径不变，nginx 零改动）并 reload；certbot 自动续期：
  ```bash
  sudo certbot certonly --standalone -d your.domain.com
  sudo cp /etc/letsencrypt/live/your.domain.com/fullchain.pem certs/server.crt
  sudo cp /etc/letsencrypt/live/your.domain.com/privkey.pem certs/server.key
  docker compose exec nginx nginx -s reload
  ```

### 第 2 步：启动应用容器（nginx + backend）

```bash
docker compose up -d
docker compose ps          # customer-service-nginx / backend 全部 Up + healthy
```

> 本 agent 只起应用容器；中间件（MySQL/Redis/Milvus/MinIO）全在共享 infra。应用启动幂等自建表 + 种子（5 个订单覆盖全场景），知识库启动幂等灌入 4 篇预置政策文档。

### 第 3 步：构建前端并访问

```bash
cd frontend
npm install
npm run build             # 生产：构建产物挂载到 nginx
# 或 npm run dev           # 开发热更新（proxy /api → backend:8000，访问 http://localhost:5173）
```

**跑起来了**：浏览器打开 **https://localhost:8443**（HTTP 8081 自动 301 到 HTTPS），用预置账号登录。浏览器首次访问自签证书会提示不受信任，点「高级 → 继续访问」即可（公网部署换正式证书后无此提示）。

| 角色 | 账号 | 密码 | 可做什么 |
|------|------|------|---------|
| 普通用户 | `user_1` | `123456` | 对话式退货/退款/投诉/查单/政策问答，预置 5 个订单 |
| 管理员 | `admin` | `.env` 的 `ADMIN_DEFAULT_PASSWORD` | 知识库管理、订单管理、一键重置测试数据 |

**访问入口**（标准端口）：

| 入口 | 地址 | 用途 |
|------|------|------|
| 前端（生产，HTTPS） | https://localhost:8443 | 唯一对外入口：TLS 终止 + 安全头；HTTP 8081 全量 301 到此 |
| 前端（生产，HTTP） | http://localhost:8081 | 自动重定向到 HTTPS（明文不承载业务） |
| 后端 API | http://localhost:8000/api/v1 | 开发调试（含 SSE）；`/metrics` Prometheus 指标同端口 |
| 前端（开发热更新） | http://localhost:5173 | Vite dev server |
| **MySQL** | `localhost:33061`（共享 infra，库 `customer_service`） | 外部工具验证数据 |
| **Redis** | `localhost:36379`（共享 infra） | 外部工具验证数据 |
| **Milvus** | `localhost:39530`（共享 infra） | 外部工具验证向量库 |

> **端口冲突**：宿主端口固定（前端 8081/8443 / 后端 8000），如需改动 `docker-compose.yml` 的 `ports` 即可；中间件端口由共享 infra 管理，本仓库不涉及。

---

## 四、使用场景与示例

### 4.1 给谁带来什么

**对用户（消费者）**
- **对话内闭环办成事**：退货、仅退款、投诉在聊天里直接完成，每一步落到数据库，退单号/退款/工单号真实可查；
- **部分退货自由指定**：任意轮次可补「只退手机壳」，确认页所见即所退，避免退错全量；
- **政策有依据**：RAG 检索知识库 → LLM 结合政策回答，空结果明确引导人工而非编造。

**对客服运营 / 管理员（`role=admin`）**
- **知识库自助维护**：文档级上传 / 编辑 / 删除 / 全量同步，MySQL 存原文、Milvus 存向量，一致性强；
- **订单数据可控**：订单 + 商品明细增删改，可模拟任意订单状态用于测试；
- **一键重置测试数据**：清空业务流水、恢复种子订单，测试无限重跑；
- **降级保障**：LLM 熔断走规则引擎 10 条正则兜底，系统永不因 AI 故障崩溃。

**对开发 / 架构**
- **可演进**：对接层 ABC 接口（`IOrderService` 等），Local → Remote 微服务 **Agent 代码零改动**；
- **可观测**：JSON 结构化日志每请求 1 进 1 出 + 6 阶段各 1 条 + LLM/DB 埋点，注入 `session_id`/`intent`/`latency_ms` 全链路追踪；
- **可验收**：SSE 契约 §5.1 事件齐备、`usage`/`done` 必选、`token` 拼接 == `done.content`（`token` 即最终回复，对接方据此取答案），契约由 `GET /api/contracts` 自动声明、可自动发现。

### 4.2 典型业务场景

预置数据（`user_1 / 123456`）可直接验证以下链路：

**场景 1 · 单轮退货闭环（业务动作真实落库）**
```
「我要退货 ORD-20240801-001」→ 回答原因 → 点确认 → 收到退货单号 RC-xxx
```
**观看点**：SSE 依次透出 `tool_call(query_order)` → `reasoning` → `token` 流式 → `usage` → `done`；admin 订单管理里商品状态已流转。

**场景 2 · 多轮部分退货（任意轮次指定商品）**
```
「我要退货」→ 追问订单号 →「ORD-20240801-001」→「只退手机壳」→ 回答原因 → 确认页只见「手机壳×1，¥29.9」
```
**观看点**：确认页商品子集与金额精确匹配；LLM 漏提取时以「只退/就退/只要退/仅退/单退 + 商品名」规则兜底，指定商品不可退时明确提示。

**场景 3 · 意图切换 + 快照恢复（多轮状态不丢）**
```
「我要退货 ORD-20240801-001」进行到一半 →「查一下订单」→ 回复「已保存退货进度，先帮您查单」→「继续退货」→ 从断点恢复
```

**场景 4 · 转人工与闲聊边界**
```
说「转人工」→ 优先触发渠道模板话术（关键词已收窄，「你是人工智能吗」不会误触发）
连续闲聊第 4 轮 → 自动收束到规则话术
```

**场景 5 · Admin 知识库与订单管理**
`admin/admin123` 登录 → 知识库文档上传/编辑/删除 → 订单增删改模拟任意状态 → 一键重置测试数据。

### 4.3 验收场景（契约声明）

`GET /api/contracts` 声明的 4 个验收场景是**行为契约**，与上述典型业务场景一一对应：

| 契约标签 | 对应业务场景 |
|---------|-------------|
| `greeting` 问候闲聊 | 场景 4 |
| `order_query` 订单查询 | 场景 3 |
| `after_sales` 售后办理（退货/退款/投诉） | 场景 1 / 2 |
| `human_handoff` 转人工 | 场景 4 |

**验收示例**（对运行中的服务）：

```bash
docker compose exec -T backend python verify_cs_e2e.py   # E2E：4 场景契约断言（token 拼接 == done.content）
```

---

## 五、技术闪光点

### 1. DeepSeek Gateway：多 Key 池化 + 排队背压 + 熔断（高并发核心）
单 Key 有 RPM 上限，撑不起海量并发——**控制并发比加连接数更重要**：
- **KeyPool**：多 Key 滑动窗口 RPM 追踪，选负载最低的 healthy Key；429 → 自动冷却，5xx → 换 Key 重试（最多 2 次，指数退避 0.1s/0.2s）；
- **排队背压**：无 healthy Key 时排队 2s，超时返回容量告警，不无限堆积；
- **熔断状态机**：连续 2 次"逻辑调用彻底失败"（重试耗尽/超时）→ 熔断 30s，期间入口直接快速拒绝、**零网络尝试**；冷却到期半开放行探测，成功即重置。超时类慢挂的代价是等满 timeout 才失败，阈值取 2 让慢挂 2×timeout 即进入冷却，用户快速转入规则引擎兜底；
- **多节点共享熔断**：熔断连续计数留本地（进程私有），冷却期信号经 Redis 广播共享——任一节点熔断，其他节点免去各自重复探测即同步降级；半开放探测成功仅广播方清除共享信号，他节点成功不 DEL（防撤销他人广播），靠 TTL 自然过期兜底；
- **故障分级**：429 全冷（AllKeysDown）与本地排队超时（CapacityExceeded）**不累计熔断**（前者已被 KeyPool 冷却覆盖，后者是本进程负载非上游故障）；已流出首个 delta 的流式中断（`StreamInterruptedError`）是连接级抖动，同样不累计——5 个并发长流自然中断不会误熔断全网关；
- **空返回兜底**：LLM 200 但 content 全空 → 固定话术兜底，且已流式/未流式两种情况都保证前端 `token` 拼接 == `done.content`；
- **usage 全计**：一轮内多次 LLM 调用（意图分类 + 生成 + 资格判定）token 用量经 contextvar 聚合，`done` 前补发单条 `usage` 事件透传对接方（观测/计费侧）。

### 2. StorageRouter：Redis 主存 + MySQL 双写自动切换（存储高可用）
- Redis 为主存储（会话/快照/缓存），MySQL 异步兜底双写；
- Redis 故障 → 自动切 MySQL 模式；后台 5s 探测 Redis，恢复后自动切回，日志记录 `storage_mode_switch`；
- MySQL 兜底读取时重建最近 10 条消息 + 摘要，保证 LLM 上下文完整。
- **数据保留（TTL 回收）**：会话/护栏判定日志默认保留 30 天（`SESSION_RETENTION_DAYS`），超期回收 MySQL 存储——后台定时 sweep（`conversation_history`/`tool_call_log` 按 `created_at` 分批删除）+ `get_session` 惰性过期（Redis miss 走 MySQL 恢复时判超期即物理回收），`delete_session` 级联清 tool_call_log。判据全在 MySQL 侧 `NOW()`（与 `created_at` 的 `CURRENT_TIMESTAMP` 同基准，无时区错位）。

### 3. RAG 一致性：MySQL 为源 + Milvus 派生索引（数据完整性）
- **根除行业通病**：纯向量库方案下 chunks 是"可变快照"，改过就无法还原原文；本项目改为 **MySQL 存原始 Markdown（source of truth），Milvus 只存分块向量快照（LlamaIndex 管理）**，每次从源重新生成，不独立演进；
- **一致性策略**：写向量失败 → 标记 `pending` → 下次写操作增量补偿（O(失败数)），全量对账仅异常恢复时 admin 手动触发；
- **检索链路**：Embedding(bge-small-zh) → Milvus Top-10 → Re-rank Top-3 → Redis 精确缓存（TTL 600s）；空结果（score<0.3）不注入 prompt，回复引导人工；
- **增量跳检**：chunks 以 `content_hash` 绑定文档与 embedding 模型，源文档/模型未变则跳过重向量化，知识库更新 O(变更) 而非 O(全量)；
- **检索可靠性**：检索服务故障 → `RetrievalUnavailableError` → LLM 常识兜底 + 低可信度声明 + 转人工建议 + 3 次/60s 冷却，不因检索挂而崩溃。

### 4. LangGraph 状态机：可完成的业务动作 + 全局可打断
- **三个业务流**（退货/仅退款/投诉）各为一张图，每轮用户输入推进一节点，停在 `awaiting`（等待输入）或 `final`（终态）；
- **全局打断**：任一节点可 `取消 / 返回 / 更换 / 转人工`；说「转人工」优先触发渠道模板话术（关键词已收窄，不误触发「人工智能」）；
- **部分退货多轮指定**：任意轮可补「只退某商品」，items 槽合并进 `return_items`，资格重算后确认页展示子集及其金额；LLM 漏提取时以「只退/就退/只要退/仅退/单退 + 商品名」句式规则兜底（含量词/语气词/多商品/否定句式处理），规避退错全量风险；
- **意图切换 + 快照**：业务流中切到别的意图自动保存当前进度（按意图各存一份），回来输入「继续退货」即可恢复；
- **模型路由**：全链路统一 `deepseek-chat`（意图分类/响应/闲聊/投诉严重性评估，200ms-1s）；退货/退款资格判定走**确定性规则**（不走 LLM）；severity 评估异常/超时降级 MEDIUM（配置字段 `deepseek_model_reasoner` 已弃用，保留作一行回退）。

### 5. SSE 契约化：事件齐备 + usage 必选 + 流式一致性（契约 §5.1）
SSE 帧按 `event: <type>\n\ndata: {json}` 格式推送，前端 `useSSE.ts` 逐帧解析：

| 事件 | 必选 | 说明 |
|------|------|------|
| `meta` | 首帧 | 接口身份声明（`agent`/`interface`/`contract_version`） |
| `status` | 可选 | 阶段进度文案（前端进度条） |
| `reasoning` | 可选 | 状态机推理依据（如投诉严重性评估依据，content+delta 双字段） |
| `tool_call` | 可选 | 工具动作透出（`query_order`/`create_return`/`create_complaint`…） |
| `token` | 流式 | `token.content`/`token.delta` 双字段逐段文本（TTFT 起点；`content`/`delta` 均为本帧增量，拼接即最终回复） |
| `usage` | **必选** | token 用量聚合（prompt/completion/total + cache 命中/未命中） |
| `done` | **必选** | 收尾，携带最终 `content` 与 `session_id` |
| `error` | 失败 | 错误文案 |

- **token 拼接 == done.content**：流式中途熔断/异常/空返回时 reply 拼接「已流部分 + 兜底」，`token` 只补发新增段，前端拼接结果与 `done.content` 严格一致；
- **content == delta 恒等**：两字段均为本帧增量（`content` 与 `delta` 同值），非累积全文——勿改为累积值，否则 token 拼接与 `done.content` 不再一致；
- **greeting 路径也补发**：新会话问候无 LLM 调用，同样补发 `token` + `usage` + `done`，契约不因降级而缺帧；
- **标准契约端点**：`GET /api/contracts` 声明 agent 的对外接口与验收场景清单（chat/login + greeting/order_query/after_sales/human_handoff），供外部系统自动发现与联调。

### 6. 并发与资源治理
- **同一会话串行锁（分布式）**：Redis 锁（SET NX PX + token Lua 释放 + 看门狗每 ttl/3 续期）串行化同会话并发请求，**多节点共享**——双击发送、多实例负载均衡下状态不覆盖；Redis 不可用 fail-fast 503、等待超时映射 429；看门狗 Redis 抖动不退出（TTL 兜底）；
- **SSE 断连自取消**：检测 `request.is_disconnected()`，自动保存进度并取消生成，防资源泄露；
- **优雅关闭**：停止接新请求 → 30s 宽限 → 保存活跃会话 checkpoint → 关闭连接池；
- **MySQL 写入重试**：连接错误/死锁指数退避（0.1/0.2/0.4s）3 次，失败统一兜底话术。

### 7. 对接层抽象：ABC 接口 + Local/Remote 实现（可演进架构）
Agent 只依赖 `IOrderService` / `IReturnService` / `IRefundService` / `IComplaintService` 等抽象接口，当前 Local 实现直连 MySQL，未来切 Remote（HTTP/gRPC 微服务）**Agent 代码零改动**。

### 8. 安全与可观测性
- **安全基线 fail-fast（启动强校验）**：`validate_security_config()` 启动即检——JWT 密钥弱（`change-me` 或 <32 字符）直接拒绝启动（无逃生开关）；admin 密码留空/`admin123` 拒绝启动（仅 `ALLOW_WEAK_ADMIN_PASSWORD=true` **且 `APP_ENV=dev`** 本地环境逃生，prod 下逃生开关强制失效，防本地便利误入生产）；
- **admin 口令 env 唯一事实来源**：`_ensure_admin_password()` 启动**无条件同步**——缺 admin 按 env 建，存量任何 hash（含历史未知弱口令）都覆盖为 env 密码；本系统无改密入口，不存在需保留的"用户改密"场景；env 密码空值再次 fail-fast 拒启（双保险）；
- **HTTPS + 安全头（对外入口 nginx）**：TLSv1.2、HSTS（max-age=1y）、`X-Frame-Options`/`X-Content-Type-Options` 防 clickjacking/MIME 嗅探，`server_tokens off` 隐藏版本；HTTP 全量 301，明文不承载业务；证书自签（`scripts/gen_cert.sh` 可重生成），公网换 Let's Encrypt；
- **/metrics（Prometheus 文本格式，零依赖）**：LLM 调用量/失败率/延迟、熔断拒绝数、排队超时、会话锁等待 429、意图规则命中率；`GET /metrics` 即可抓取，接入告警（如失败率飙升/熔断 opens）；
- **三层 Prompt Injection 防护**：预处理正则（零成本拦截）+ prompt 五维度法（结构、边界、权限、数据、应急）+ System Prompt 安全注入；
- **JWT 认证**：payload 含 role，Admin 接口权限隔离，前端按 role 显隐入口；
- **JSON 结构化日志**：每个请求 1 进 1 出 + 6 阶段各 1 条 + LLM 调用/DB 操作全埋点，注入 `session_id` / `intent` / `latency_ms`，支持全链路追踪；
- **会话消息体截断**：`SESSION_MAX_MESSAGES`（默认 40 条）超限截断为「首条 user 消息 + 最近 N-1 条」，防止会话无限增长（LLM/状态机不读消息全文，截断仅影响前端历史展示）。

### 9. 工程化质量
- **后端 386 项测试全绿**：意图分类（6 类准确率 >90%）、退货/退款/投诉状态机（7 节点 + 三级判定）、编排器（多轮商品合并、确认/取消 action 语义、转人工优先、流式一致性、空返回兜底）、LLM 工具决策循环（护栏 allow/reject/override 三态、同参去重、累计调用截断）、SSE 契约（帧格式/usage 必选/token-done 一致）、DeepSeek Gateway（熔断 open/半开放行/成功 reset、共享熔断 close 仅本地广播方、chat 与 chat_stream 双入口、流中断隔离、429/5xx/超时退避、首 delta 后不重试、兜底异常元组去重）、部分退货规则兜底、RAG、StorageRouter、分布式锁（获取/等待/超时/看门狗续期）、tool_call_log 落库、应用启动自建表+种子（schema 幂等）、admin 口令兜底（创建/弱口令 sync/强口令不覆盖）、安全基线校验（弱 JWT/弱 admin 拒绝、逃生放行）、/metrics 埋点（计数器/Summary/格式）、会话历史/消息截断、contracts 端点；
- **前端 41 项测试全绿**：ChatPanel 渲染、ChatInput、登录/注册表单、useChat/useSession/useSSE、formatTime、客服/登录/注册视图；
- **集成测试可重跑**：真实服务链路（会话 → SSE → 退单落库），服务未启动自动跳过不误报。

### 10. LLM 工具决策循环 + 实时护栏（P3-P5）
- **决策循环**：`ORDER_STATUS` / `POLICY_INQUIRY` 意图下，LLM 自主决定调只读工具（`search_policy` / `query_order` / `list_user_orders`），工具结果回灌后由生成节点组装回复；业务副作用工具（建退货/退款/投诉单、资格判定）的决策被护栏拦截，转由 LangGraph 状态机确定性接手——决策与执行分离，LLM 不裸调业务动作；
- **规则短路（优化③）**：`ORDER_STATUS` 且输入命中订单号时跳过 LLM 决策、确定性直查 `query_order`（未命中订单连查 `list_user_orders` 兜底），零决策轮调用、单号提取 100% 准确；`POLICY_INQUIRY` 刻意不接管（检索 query 依赖 LLM 改写）；未命中/异常回退 LLM 决策循环；落库 `verdict=rule_shortcut` 可观测；
- **实时护栏 ToolGuardrail**：决策与执行之间的确定性规则校验，输出 `allow / reject / override` 三态 + 机器可读理由——副作用工具 reject→business、`search_policy` 过短/纯问候 reject、`query_order` 缺单号 override 为列最近订单、同轮同参数 dedupe 复用首次结果、累计工具调用 >3 截断强制出路由；
- **观测落库**：每次护栏判定写 `tool_call_log`（session / round / tool / verdict / reason / 结果摘要 / 延迟），落库失败静默不阻断决策；为管理侧调用分析预留数据底座。

### 11. 意图分类规则前置短路（少调 LLM）
- **少调用优于小模型**：DeepSeek 官方仅 deepseek-chat/reasoner 两档，优化空间在"少调用"——意图分类每轮必调 LLM，但问候/查单/退货/退款/投诉等大量 query 是模板化表达，新增正则判定层（`intent_rules.py`）在 `classify_intent` 开头确定性接管，未命中回退 LLM，标准负载下意图分类 LLM 调用估降 40-60%（成本 + 首字延迟双降）；
- **保守接管**：只接管正则可锁死的模式；疑问句式（"能退货吗" vs "我要退货"）一律回退 LLM（政策/资格咨询语义）；**POLICY_INQUIRY 刻意不接管**（政策问法最复杂，保分类准确率优先）；规则命中置信度 0.97 且 usage=None（聚合器安全跳过）；
- **安全边界**：业务流内（确认/好的/补充等短词）与注入命中强制禁用规则——短词保留 LLM+state_hint、注入保留防御声明；
- **可观测**：命中打 `event=intent_rule_hit`（intent + 输入长度），上线可量化命中率与 LLM 调用降幅。

---

## 六、技术栈一览

| 层 | 技术 | 说明 |
|----|------|------|
| 后端 | Python 3.11 + FastAPI | async/await，SSE 流式，OpenAPI 自动文档 |
| 状态编排 | LangGraph | 三业务流状态机，每轮输入推进一节点 |
| 前端 | Vue3 + Vite + Element Plus + Pinia | 客服聊天 + Admin 管理后台，SSE 逐帧消费 |
| 关系数据库 | MySQL 8 | 业务权威数据 + 知识库原文源（source of truth） |
| 缓存/会话 | Redis 7 | 会话主存 / 快照 / RAG 精确缓存 / 分布式锁 / 熔断冷却共享信号 |
| 向量库 | Milvus + LlamaIndex + bge-small-zh | 知识库派生向量索引，Top-10 → Re-rank Top-3 |
| LLM | DeepSeek（openai 兼容） | 统一 chat（意图/响应/闲聊/严重性）；资格判定规则化，超时降级 |
| 网关 | 自研 KeyPool + 熔断 | 多 Key 滑动窗口 RPM + 排队背压 + 指数退避 + 规则引擎兜底 |
| 测试 | pytest + pytest-asyncio + vitest | 后端 386 / 前端 41，集成测试真实链路可重跑 |

---

## 七、配置说明

关键环境变量（完整见 `.env.example`，每项含注释）：

| 配置 | 说明 | 默认值 |
|------|------|--------|
| `DEEPSEEK_API_KEYS` | DeepSeek API Key 列表，逗号分隔（**必填**，Key 越多并发越高） | - |
| `DEEPSEEK_BASE_URL` | API 地址 | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL_CHAT` | chat 模型（意图/响应/闲聊/严重性） | `deepseek-chat` |
| `DEEPSEEK_MODEL_REASONER` | ~~reasoner 模型~~ 已弃用（投诉严重性改走 chat，字段保留作一行回退） | `deepseek-reasoner` |
| `DEEPSEEK_PER_KEY_RPM` | 单 Key 每分钟请求上限（滑动窗口追踪） | `200` |
| `DEEPSEEK_QUEUE_MAX_SIZE` / `DEEPSEEK_QUEUE_TIMEOUT` | 排队容量 / 排队超时（秒，超时返回容量告警） | `500` / `2.0` |
| `DEEPSEEK_TIMEOUT_CHAT` / `DEEPSEEK_TIMEOUT_REASONER` | chat 调用超时（秒）/ ~~reasoner 超时~~ 已弃用（随 reasoner 字段一并保留作回退） | `8.0` / `15.0` |
| `SERVICE_MODE` / `VECTOR_STORE` | 运行模式 / 向量库实现 | `local` / `milvus` |
| `REDIS_URL` | 共享 Redis 连接（db index 1） | `redis://redis:6379/1` |
| `MYSQL_URL` | MySQL 连接（asyncmy 异步驱动，密码仅 env 注入） | `mysql+asyncmy://csuser:CHANGE_ME@mysql:3306/customer_service` |
| `MYSQL_POOL_SIZE` | 连接池大小（asyncmy 池上限） | `20` |
| `JWT_SECRET_KEY` | JWT 签名密钥（生产必须替换为随机长字符串） | `change-me...` |
| `JWT_EXPIRE_HOURS` | token 有效期（小时） | `2` |
| `SESSION_TTL` | 会话 TTL（秒） | `3600` |
| `SESSION_LOCK_TTL` | 会话分布式锁 TTL（秒，看门狗每 ttl/3 续期防长处理击穿） | `60` |
| `SESSION_LOCK_WAIT_TIMEOUT` | 获取锁最大等待（秒），超时映射 429 | `30` |
| `SESSION_LOCK_POLL_INTERVAL` | 抢锁失败轮询间隔（秒） | `0.1` |
| `CONVERSATION_MAX_ROUNDS` | 对话最大轮次 | `10` |
| `SESSION_RETENTION_DAYS` | 会话/工具判定日志保留天数，超期定时+惰性回收（回收 MySQL 存储） | `30` |
| `SESSION_CLEANUP_INTERVAL_SECONDS` | 定时清理周期（秒） | `3600` |
| `SESSION_CLEANUP_BATCH_SIZE` | 单批删除行数（控制事务大小） | `500` |
| `MILVUS_URI` / `MILVUS_COLLECTION` | Milvus 连接 / collection（共享 Milvus 加 `cs_` 前缀隔离） | `http://milvus:19530` / `cs_knowledge` |
| `RAG_CACHE_TTL` / `INTENT_CACHE_TTL` | RAG 检索缓存 / 意图分类缓存 TTL（秒） | `600` / `60` |
| `APP_ENV` | 部署环境 `dev` / `prod`。**生产必设 prod**：prod 下弱口令逃生开关强制失效 | `dev` |
| `ADMIN_DEFAULT_USERNAME` / `ADMIN_DEFAULT_PASSWORD` | 管理端账号（密码由 env 注入；留空或弱口令时**启动 fail-fast** 拒绝启动；启动时**无条件同步**到数据库，env 是唯一事实来源） | `admin` / 无默认 |
| `ALLOW_WEAK_ADMIN_PASSWORD` | 本地环境逃生开关：`true` 允许弱 admin 口令启动（**仅 `APP_ENV=dev` 有效**，prod 下设了也拒绝） | `false` |

---

## 八、目录结构

```
customer-service/
├── backend/                      # 后端源码（FastAPI）
│   ├── app/
│   │   ├── main.py               # 应用入口 + 生命周期（优雅关闭）
│   │   ├── api/                  # 交互层：REST 路由 + SSE（routes / auth / deps / contracts）
│   │   ├── agent/                # 控制层：Agent 编排
│   │   │   ├── orchestrator.py   # 6 阶段流水线 + SSE 事件发射 + 熔断兜底
│   │   │   ├── intent.py         # 意图分类（6 类：退货/退款/投诉/查单/政策/闲聊）
│   │   │   ├── rule_engine.py    # 规则引擎兜底（10 条正则，LLM 熔断时生效）
│   │   │   ├── usage.py          # token 用量聚合（contextvar，按 asyncio task 隔离）
│   │   │   ├── state_machine/    # 退货/退款/投诉状态机（LangGraph）
│   │   │   ├── function_calling/ # 工具 + 护栏（order/return/refund/policy tools、guardrail、tool_call_log）
│   │   │   └── prompts/          # prompt 模板（意图/闲聊/政策…）
│   │   ├── infrastructure/       # 资源层门面：interfaces（Protocol）+ 统一导出（llm_gateway/mysql_pool/retriever）+ DeepSeek Gateway + MySQL + schema
│   │   ├── rag/                  # 资源层：RAG（embedder/retriever/milvus_impl/kb_store/knowledge）
│   │   ├── services/             # 能力层：业务服务（interfaces ABC + local_impl + 重试）
│   │   ├── session/              # 资源层：会话管理（Redis 主存 + MySQL 兜底 + 消息截断 + 分布式锁）
│   │   └── utils/
│   ├── sql/init.sql              # 建表 + 种子数据
│   └── tests/                    # 380+ 项单元/契约/集成测试
├── frontend/                     # 前端（Vue3 + Vite + Element Plus）
│   ├── src/
│   │   ├── api/                  # axios 接口模块
│   │   ├── components/           # chat 组件（ChatInput/ChatPanel 等）
│   │   ├── composables/          # useChat / useSession / useSSE
│   │   ├── stores/               # authStore（JWT + 用户信息）
│   │   ├── views/                # 登录/注册/客服聊天/管理后台
│   │   └── __tests__/            # 41 项组件/组合式单测
│   └── vitest.config.ts
├── docker/                       # Dockerfile.backend / nginx.conf
├── docs/
├── docker-compose.yml            # 全栈容器编排（一键启动）
├── .env.example                  # 环境变量模板（每项含注释）
├── solution.md                   # 技术方案（设计依据）
├── task.md                       # 任务拆分与验收标准
└── README.md
```

---

## 九、测试与验收

| 阶段 | 内容 | 结果 |
|------|------|------|
| 后端单元/契约 | 意图（含规则前置短路）/ 状态机 / 编排器 / 决策循环（含规则短路）+护栏 / SSE 契约 / Gateway 熔断+退避 / usage / RAG / 会话 / 分布式锁 / tool_call_log / contracts / 建表种子 / TTL 清理 / severity 模型档守护 | **383 passed** |
| 前端组件 | ChatPanel / ChatInput / 登录注册表单 / useChat / useSession / useSSE / formatTime / 视图 | **41 passed** |
| 集成测试 | 真实服务链路（会话 → SSE → 退单落库），`GET /healthz` 探测，未启动自动跳过 | 服务在跑时 **3 passed** |
| E2E | `backend/verify_cs_e2e.py`：4 场景契约断言（闲聊/订单/政策/投诉，token 拼接 == done.content） | 已验证通过 |

运行全部测试：

```bash
# 后端（推荐容器内执行，依赖与 .env 连接串已就绪）
docker compose exec backend python -m pytest tests/ -q

# 后端（本地执行需先装 backend/requirements.txt，并把 .env 中
# REDIS_URL/MYSQL_URL/MILVUS_URI 改为本机可达；如遇 pytest_html 缺 py 包报错，加 -p no:html）
cd backend && python -m pytest tests/ -q -p no:html

# 前端（需在 frontend 目录下执行，vitest 才能解析 @ 别名）
cd frontend && npx vitest run
```

---

## 十、开发指南

### 环境

```bash
cp .env.example .env              # 配 DEEPSEEK_API_KEYS / JWT_SECRET_KEY
docker compose up -d              # 起应用容器（中间件全在共享 infra）
docker compose exec backend bash  # 进入后端容器开发/调试
```

### 改后端代码
- **backend 容器无源码挂载**：改 `backend/` 代码后需 `docker compose build backend && docker compose up -d backend` 重建镜像才生效（本地想热重载则在本机 `uvicorn app.main:app --reload`）；
- 改 `.env` 配置后 `docker compose up -d` 重启容器即可。

### 跑测试 / 验收
- 提交前先跑后端 `pytest tests/ -q` 与前端 `vitest run`，确保不破坏既有 380+ + 41 项；
- 集成测试需服务在跑（`docker compose up -d`），未启动自动跳过不误报。

### 新增 API
- **契约优先**：models → services → api 路由 → 注册到 `main.py`；对外接口先出方案，评审确认后再实现；
- 新增 SSE 事件类型时，同步更新 `backend/app/api/contracts.py` 的 MANIFEST 与前端 `useSSE.ts` 的事件分支。

### 编码规范（约定）
- 4 空格缩进、阿里 Java 规范思想；注释写"为什么"不写"做什么"（public 方法 Javadoc 除外）；中文注释、英文标识符；
- 新增接口/消费者打印入参出参（debug 级）；核心逻辑（Service 业务分支含正/异常路径）必须单测覆盖。

---

## 十一、常见问题

| 现象 | 处理 |
|------|------|
| 首次启动慢 | MySQL 建表灌种子、backend 首次加载 embedding 模型（bge-small-zh）需要时间，等 `docker compose ps` 全 healthy 再访问 |
| 没有 DeepSeek Key | `DEEPSEEK_API_KEYS` 留空时系统可正常启动，所有 LLM 调用走规则引擎兜底（仅能回复预置话术） |
| 端口冲突 8081/8000 | 修改 `docker-compose.yml` 对应 `ports`（中间件端口由共享 infra 管理，不涉及） |
| 改了 backend 代码不生效 | backend 容器无源码挂载，需 `docker compose build backend && docker compose up -d backend` 重建 |
| 本地 pytest 报 pytest_html 缺 py 包 | 加 `-p no:html`：`python -m pytest tests/ -q -p no:html` |
| 前端白屏 | `frontend/dist` 未构建：`cd frontend && npm run build`，重启 nginx |
| 想重置测试数据 | admin 登录 → 管理后台 → 订单 tab → 「重置测试数据」，一键恢复种子订单 |
| 提示"系统繁忙" | 多 Key 全冷却 / LLM 排队超时（`DEEPSEEK_QUEUE_TIMEOUT`）/ 熔断冷却期，属背压与降级的正常表现，稍后重试 |

---

## 十二、已知限制与优化方向

**如实说明当前已知的性能与边界问题：**

1. **模型全 CPU 推理**：bge-small-zh embedding、DeepSeek chat 判定在 CPU/远程 API 上运行，首次加载 embedding 模型较慢（启动已预热，重启后首次提问前加载）。重负载场景可考虑 GPU 容器。
2. **RAG 缓存仅精确命中**：Redis 缓存 key 为精确问题文本（TTL 600s），语义相近的问法走完整检索 + LLM 调用（DeepSeek 计费）。高频率场景可评估语义级缓存。
3. **规则引擎兜底覆盖面有限**：10 条正则仅覆盖高频售后场景（退货/退款/订单/政策/投诉），LLM 故障时的自由问答只能返回固定话术，无法真正理解。
4. **回合缓存仅限无状态政策轮次**：订单/业务流等依赖实时数据的轮次不缓存（正确性优先），缓存命中率受限于政策问答占比。
5. **SSE 断连不重放已流内容**：流式中途断线会保存进度并取消生成，但已流出的内容不会自动补发，用户需重新进入会话查看历史。

---

## 十三、版本记录

| 版本 | 日期 | 核心内容 |
|------|------|----------|
| **2.4.3** | 2026-08-28 | 资源层门面：infrastructure 统一门面（interfaces 4 Protocol + re-export 单例 llm_gateway/mysql_pool/retriever），agent 7 文件改依赖抽象不依赖实现，retriever 惰性导出破除循环导入，新增门面身份测试；README 架构图重构为四层（交互/控制/能力/资源）+ 职责表 + 目录分层标注，措辞专业化，测试数字统一（386/383/41） |
| **2.4.2** | 2026-08-28 | 上线前安全加固三件套：JWT/admin 弱口令 fail-fast 拒绝启动（`ALLOW_WEAK_ADMIN_PASSWORD` 逃生开关仅 dev 生效、生产强制失效）、全站 HTTPS + 自签证书 + 80→443 跳转、Prometheus `/metrics` 指标导出；GitHub Actions CI 门禁（backend pytest + frontend vitest/build，dev/main/PR 全触发）；多副本状态外置实测：compose backend 去固定容器名/宿主端口（静态端口与 `--scale` 冲突）、原容器名改网络别名 + 共享网关 DNS 多 IP 自动轮询（网关配置零改动），`verify_multinode.py` 四层验证（部署可达 / 锁互斥跨进程 / 端到端并发 / 熔断广播）实测 11 PASS；测试收敛 CI 环境隐式依赖（BGE tokenizer 全局 mock、admin 密码测试自包含 mock） |
| **2.4.0** | 2026-08-26 | 统一 API 网关接入：前端 nginx 改反代共享网关 `api-gateway:8099`（Host: cs.local），网关负责 X-Request-ID traceId 根生成（后端日志 `trace_id` 对齐）、按真实 IP 限流（cs_chat 2r/s + cs_auth 5r/m 接管登录限流）、SSE 透传；DeepSeek thinking 参数化（意图分类/投诉评估关闭省思考 token）+ reasoning 事件全链路展示（决策非流式全文 / 生成流式增量）+ 前端思考/来源折叠；检索来源 `[来源N]` 上下文序号化 + 前端来源内容展示；测试扩充至 369 项 |
| **2.3.0** | 2026-08-26 | 多节点状态外置（自由扩缩容）：per-session Redis 分布式锁（SET NX PX + token Lua 释放 + 看门狗续期 + Redis 抖动容忍，替代进程内 asyncio.Lock）；DB/LLM/KB 熔断计数留本地、冷却信号 Redis 广播共享（close 仅本地广播方生效防撤销他人广播）；Milvus 同步检索走 to_thread 不阻塞事件循环/锁看门狗；StorageRouter Redis key 失效回退 MySQL；Redis 不可用 fail-fast 503、锁等待超时 429；测试扩充至 361 项 |
| **2.2.6** | 2026-08-26 | 决策循环规则短路（优化③）：`ORDER_STATUS` 命中订单号跳过 LLM 决策、确定性直查 `query_order`，未命中订单连查 `list_user_orders` 兜底——单号提取 100% 准确 + 零决策轮调用；`POLICY_INQUIRY` 刻意不接管（检索 query 依赖 LLM 改写）、未命中/异常回退 LLM 决策循环；落库 `verdict=rule_shortcut` 可观测；测试扩充至 351 项 |
| **2.2.5** | 2026-08-26 | 投诉严重性评估 reasoner→chat（降本增效）：全项目唯一 LLM 档收敛到 chat，reasoner 配置保留作一行回退；prompt 增强校准（物流时效归 MEDIUM、安全类列举、LOW 收紧），实测 17/17=100% 与 reasoner 打平切换无损，HIGH 判据（人身安全/批量/金额>5000）零漏判；新增 verify_severity_accuracy.py 准确性基线；测试扩充至 347 项 |
| **2.2.4** | 2026-08-26 | 意图分类规则前置短路（少调 LLM）：新建 intent_rules 正则层，高置信模板化表达（问候/查单/退货/退款/投诉）跳过 LLM 分类、未命中回退；POLICY_INQUIRY 刻意不接管、业务流内与注入命中强制禁用规则；命中打 `event=intent_rule_hit` 可观测；测试扩充至 341 项 |
| **2.2.3** | 2026-08-26 | 会话数据 TTL 清理（回收 MySQL 存储）：conversation_history/tool_call_log 保留 30 天超期回收，后台定时分批 sweep + get_session 惰性过期，delete_session 级联清 tool_call_log，补 idx_created_at 索引；SSE 内容帧对齐统一契约（answer→token，content+delta 双字段）；测试扩充至 309 项 |
| **2.2.2** | 2026-08-26 | LLM 网关熔断 + 换 Key 重试退避 + 流中断隔离 + 空返回兜底 + 兜底异常元组去重；管理端文件上传（覆盖更新复用 upsert）；异常治理收尾（写路径幂等、检索冷却、DB 熔断）；RAG 增量跳检（content_hash）+ 章节级检索扩充；测试扩充至 297 项 |
| **2.2.1** | 2026-08-26 | prompt 五维度法防注入（三层防护 + system prompt 结构化）；FC 契约优化（全量 `{ok,data,error}` 信封 + schema 规范化 + query 清洗）；分块前文本清洗 + 检索 query 归一化 |
| **2.2** | 2026-08-26 | P6 回合缓存：无状态政策轮次短路整图复用答案（`search_policy.ok` 门控）；标题层级切分 + 章节级检索扩充 |
| **2.1** | 2026-08-25 | P3.3 切换共享 infra：应用启动幂等自建表 + 种子，去除自带中间件（MySQL/Redis/Milvus 全走共享 infra） |
| **2.0** | 2026-08-24 | P1 SSE 契约对齐 + 转人工优先；Milvus 迁移（去 chroma，LlamaIndex 托管）；顶层 6 阶段流水线图化为 LangGraph；P3 工具决策循环；P4 实时护栏 ToolGuardrail；P5 护栏判定落库 tool_call_log |
| **1.1** | 2026-08-07 | 会话能力补齐：对话历史（会话列表/读取/删除 + 前端多会话侧边栏）、消息体截断、MySQL 兜底 datetime 序列化修复；部分退货多轮指定 + items 规则兜底 |
| **1.0** | 2026-08-07 | 高并发 AI Agent 智能客服系统完整实现：三业务流状态机（退货/退款/投诉）+ RAG 检索 + DeepSeek 多 Key 网关 + StorageRouter 双写 |

---

## 附录：状态机数据契约（LangGraph State）

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
| `return_items` | list | 任意推进轮（仅退货） | 用户指定只退的部分商品名数组；多轮可补充合并（LLM 提取 + 规则兜底），空=退全部可退商品 |
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

**执行模型**（每轮循环推进到等待输入或终态为止）：

```
第 N 轮用户输入
  → 注入 user_input 进 state
  → 按 state.stage 路由到目标节点并执行，返回部分更新（LangGraph 自动浅合并）
  → 若节点未设置 awaiting/final，继续路由下一节点（同一轮可连续推进多个节点）
  → 停在 awaiting（等待输入）或 final（终态）则本轮结束，等待下一轮
```

以退货流程为例，state 随对话逐节点变化：

| 用户输入 | 经过节点 | stage | awaiting | 关键变化 |
|---------|---------|-------|----------|---------|
| 「我要退货 ORD-...」 | collect_order_id → verify_order | verify_order | — | 提取 order_id |
| （同轮继续） | check_eligibility | collect_reason | reason | 写入 order、eligibility |
| 「质量问题」 | collect_reason | confirm | confirm | 写入 reason |
| 「确认」 | confirm → execute → notify | END | — | 写入 result，final=True |

---

## 文档索引

- **技术方案**：[solution.md](solution.md)（架构设计、数据模型、API 契约、风险控制）
- **任务拆分与验收**：[task.md](task.md)（逐项任务与验收标准）
