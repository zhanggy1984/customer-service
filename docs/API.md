# API 文档

Base URL: `http://localhost:8000/api/v1`（经 nginx 为 `http://localhost/api/v1`）

## 认证

### 注册
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username":"demo","password":"demo123","phone":"13800000000"}'
# 201 {"msg":"注册成功"}
```

### 登录
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"user_1","password":"123456"}'
# 200 {"access_token":"eyJ...","token_type":"bearer","user":{"id":2,"username":"user_1","role":"user",...}}
```
JWT payload 含 `sub`/`username`/`role`，有效期默认 2h。后续请求头带 `Authorization: Bearer <token>`。

## 会话与消息

### 创建会话
```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Authorization: Bearer <token>"
# 201 {"session_id":"a1b2c3..."}
```

### 发送消息（SSE 流式）
```bash
curl -N -X POST http://localhost:8000/api/v1/sessions/{session_id}/messages \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"content":"我要退货 ORD-20240801-001"}'
```
响应为 `text/event-stream`：
```
data: {"type":"status","stage":"preprocess","message":"正在处理您的问题..."}
data: {"type":"status","stage":"intent","message":"正在理解您的问题..."}
data: {"type":"action","action":"confirm","intent":"RETURN_REQUEST"}   # 确认节点时
data: {"type":"done","intent":"RETURN_REQUEST","content":"退货单 RC-... 已创建","session_id":"..."}
data: {"type":"error","message":"..."}   # 出错时
```

会话不存在/过期时自动重建，`done.session_id` 为实际会话 id。

### 会话历史（侧边栏列表 + 历史消息 + 删除）

```bash
# 当前用户的会话列表（空会话不展示，按最近写入倒序，最多 50 条）
GET /sessions
# 200 {"items":[{"session_id":"a1b2...","title":"我要退货 ORD-001","updated_at":"2026-08-07T12:00:00","intent":"RETURN_REQUEST"}],"total":1}
#   title = 首条 user 消息前 30 字；assistant 开头/无 user 消息的会话标题为"新会话"
#   updated_at = 最后保存/活跃时间（conversation_history 每次保存 DELETE+INSERT 刷新，非会话创建时间）

# 拉取单个会话的历史消息（归属校验；Redis 过期自动走 MySQL 恢复）
GET /sessions/{sid}/messages
# 200 {"session_id":"a1b2...","intent":"RETURN_REQUEST","messages":[{"role":"user","content":"...","ts":"2026-08-07T12:00:00Z"}]}
# 403 无权访问他人会话 / 404 会话不存在

# 删除单个会话（Redis + MySQL conversation_history 一并清除，仅限本人）
DELETE /sessions/{sid}
# 200 {"msg":"已删除"} / 403 / 404
```

前端将当前会话 `session_id` 持久化到 localStorage（按用户隔离，key `cs_session_{user_id}`），刷新页面 / 重新登录后自动恢复最近会话并加载历史，可继续对话；侧边栏支持切换历史会话与删除。

## Admin（需 `role=admin`）

### 知识库管理
```bash
# 列表
GET /admin/knowledge
# 上传（Markdown）
POST /admin/knowledge
#  body: {"title":"运费政策","content":"# 政策内容..."}
# 删除单文档
DELETE /admin/knowledge/{doc_id}
```

### 订单管理
```bash
# 列表（含商品明细）
GET /admin/orders
# 创建订单（含商品明细）
POST /admin/orders
#  body: {"order_id":"ORD-...","user_id":2,"status":"PAID","total_amount":89.85,
#         "shipping_address":"...","items":[{"item_id":"SKU-9","name":"数据线","price":29.95,"quantity":2,"returnable":true}]}
# 修改订单（status/total_amount/shipping_address）
PUT /admin/orders/{order_id}
# 删除订单
DELETE /admin/orders/{order_id}
# 添加商品明细
POST /admin/orders/{order_id}/items
# 删除商品明细
DELETE /admin/orders/{order_id}/items/{item_db_id}
```

## 健康检查

```bash
GET /healthz   # {"status":"ok"}
```

## 核心 Agent 意图

| 意图 | 触发示例 | 行为 |
|------|---------|------|
| RETURN_REQUEST | "我要退货 ORD-001" | 状态机：验证→资格→原因→确认→创建退单 |
| REFUND_REQUEST | "我想仅退款 ORD-003" | 三级判定：PAID 可退 / SHIPPED 拒收 / DELIVERED 走退货 |
| COMPLAINT | "客服态度差我要投诉" | reasoner 评估严重性 → 创建工单 |
| ORDER_STATUS | "查一下订单 ORD-001" | 查单 + 状态，缺单号时列出最近订单 |
| POLICY_INQUIRY | "退货时限是多久" | RAG 检索政策 → LLM 回答 |
| CHITCHAT | "你好" | LLM 自由回复，第 4 轮规则话术兜底 |

支持业务中途切换意图（快照保存 + 恢复），输入"取消"可中断任意流程。
