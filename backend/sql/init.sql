-- =============================================================
-- 智能客服系统 — MySQL 初始化脚本
-- 由 docker-compose 挂载到 mysql 容器的 /docker-entrypoint-initdb.d/
-- 仅在数据卷为空(首次启动)时执行一次。改动表结构后需 docker compose down -v 重建。
-- 注意: 数据库名需与 docker-compose.yml 的 MYSQL_DATABASE 保持一致。
-- =============================================================
SET NAMES utf8mb4;
USE customer_service;

-- ---------- users ----------
CREATE TABLE IF NOT EXISTS users (
    id            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username      VARCHAR(64)  NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(16)  NOT NULL DEFAULT 'user',   -- admin | user
    phone         VARCHAR(20)  NULL,
    created_at    DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- orders ----------
CREATE TABLE IF NOT EXISTS orders (
    id               BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_id         VARCHAR(32)  NOT NULL UNIQUE,
    user_id          BIGINT UNSIGNED NOT NULL,
    status           VARCHAR(20)  NOT NULL,               -- PAID/SHIPPED/DELIVERED/CANCELLED
    total_amount     DECIMAL(10, 2) NOT NULL,
    shipping_address VARCHAR(255) NULL,
    created_at       DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    delivered_at     DATETIME     NULL,
    KEY idx_user (user_id),
    CONSTRAINT fk_orders_user FOREIGN KEY (user_id) REFERENCES users (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- order_items（支持部分退货）----------
CREATE TABLE IF NOT EXISTS order_items (
    id        BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    order_id  BIGINT UNSIGNED NOT NULL,
    item_id   VARCHAR(32)  NOT NULL,
    name      VARCHAR(128) NOT NULL,
    price     DECIMAL(10, 2) NOT NULL,
    quantity  INT          NOT NULL DEFAULT 1,
    returnable TINYINT(1)  NOT NULL DEFAULT 1,            -- 0=定制/不可退
    status    ENUM('NORMAL', 'RETURN_REQUESTED', 'RETURNED', 'REFUNDED')
              NOT NULL DEFAULT 'NORMAL',
    KEY idx_order (order_id),
    CONSTRAINT fk_items_order FOREIGN KEY (order_id) REFERENCES orders (id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- return_orders ----------
-- UNIQUE(order_id, user_id): 防止同一用户对同一订单并发重复创建退货单
CREATE TABLE IF NOT EXISTS return_orders (
    id           BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    return_id    VARCHAR(32)  NOT NULL UNIQUE,
    order_id     BIGINT UNSIGNED NOT NULL,
    user_id      BIGINT UNSIGNED NOT NULL,
    items        JSON         NOT NULL,                   -- [{"item_id":..,"name":..,"quantity":..,"refund":..}]
    reason       VARCHAR(255) NULL,
    refund_amount DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    status       VARCHAR(16)  NOT NULL DEFAULT 'APPROVED',
    session_id   VARCHAR(64)  NULL,
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_return_order_user (order_id, user_id),
    KEY idx_user (user_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- refund_orders ----------
CREATE TABLE IF NOT EXISTS refund_orders (
    id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    refund_id  VARCHAR(32)  NOT NULL UNIQUE,
    order_id   BIGINT UNSIGNED NOT NULL,
    user_id    BIGINT UNSIGNED NOT NULL,
    reason     VARCHAR(255) NULL,
    amount     DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
    status     VARCHAR(16)  NOT NULL DEFAULT 'APPROVED',
    session_id VARCHAR(64)  NULL,
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user (user_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- complaint_tickets ----------
CREATE TABLE IF NOT EXISTS complaint_tickets (
    id             BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    ticket_id      VARCHAR(32)  NOT NULL UNIQUE,
    user_id        BIGINT UNSIGNED NOT NULL,
    order_id       VARCHAR(32)  NULL,
    complaint_type VARCHAR(32)  NULL,
    description    TEXT         NULL,
    severity       VARCHAR(16)  NULL,                     -- HIGH/MEDIUM/LOW
    status         VARCHAR(16)  NOT NULL DEFAULT 'OPEN',
    session_id     VARCHAR(64)  NULL,
    created_at     DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_user (user_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- knowledge_docs（知识库文档源数据）----------
-- MySQL 是知识库单一事实来源（source of truth），ChromaDB 只存分块向量快照（派生）。
-- sync_status: ok=已同步 / pending=ChromaDB 写入失败待补偿（admin 重试或全量对账自愈）
CREATE TABLE IF NOT EXISTS knowledge_docs (
    id          BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    source      VARCHAR(100) NOT NULL UNIQUE,   -- 文档标题，也是 ChromaDB metadata.source 关联键
    content     MEDIUMTEXT   NOT NULL,          -- 原始 Markdown 全文
    updated_by  VARCHAR(64)  NULL,
    sync_status VARCHAR(16)  NOT NULL DEFAULT 'ok',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- conversation_history（会话异步快照兜底）----------
CREATE TABLE IF NOT EXISTS conversation_history (
    id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    user_id    BIGINT UNSIGNED NOT NULL,
    intent     VARCHAR(32)  NULL,
    messages   JSON         NULL,
    agent_state JSON        NULL,
    summary    TEXT         NULL,
    result     TEXT         NULL,
    created_at DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_session (session_id),
    KEY idx_user (user_id)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- ---------- tool_call_log（P5：护栏判定观测）----------
-- 决策轮每次 guardrail.check 判定落一条（含 allow/reject/override/dedupe）。
-- args_json 存原始入参；result_summary 存结果摘要（前 N 字 + 命中数），空串=工具空结果。
CREATE TABLE IF NOT EXISTS tool_call_log (
    id             BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id     VARCHAR(64) NOT NULL,
    user_id        INT         NOT NULL,
    round_no       INT         NOT NULL,              -- agent_loop 内第几轮
    tool_name      VARCHAR(64) NOT NULL,              -- override 时记最终执行工具
    args_json      JSON        NULL,
    result_summary VARCHAR(512) NULL,                 -- 空 = 工具返回空结果（search 无命中/not_found）
    latency_ms     INT         NULL,                  -- 真执行工具耗时；reject/dedupe 记 0
    verdict        VARCHAR(16) NOT NULL,              -- allow / reject / override
    verdict_reason VARCHAR(255) NULL,                 -- side_effect/trivial_query/dedupe/...
    llm_confidence FLOAT       NULL,                  -- 暂无置信度来源，预留
    query_text     VARCHAR(500) NULL,                 -- 本轮用户输入，便于请求分析
    created_at     DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    KEY idx_session (session_id),
    KEY idx_tool_time (tool_name, created_at)
) ENGINE = InnoDB DEFAULT CHARSET = utf8mb4;

-- =============================================================
-- 种子数据
-- =============================================================
-- 密码: admin/admin123, user_1/user_2 均为 123456（bcrypt 预生成 hash）
INSERT INTO users (username, password_hash, role, phone) VALUES
    ('admin',  '$2b$12$K3vOmcMr0lF8hKB1.tfecu.22mtv6RK01l7B/.eR4kgryCclvflSW', 'admin', '13800000000'),
    ('user_1', '$2b$12$jl38CVX4S2zf2sdwxweUyuN/WMk2zbCifYtx.EPJfnxUyLAmTDvdC', 'user',  '13800000001'),
    ('user_2', '$2b$12$jl38CVX4S2zf2sdwxweUyuN/WMk2zbCifYtx.EPJfnxUyLAmTDvdC', 'user',  '13800000002');

-- 订单（user_1 -> id=2, user_2 -> id=3）
INSERT INTO orders (order_id, user_id, status, total_amount, shipping_address, created_at, delivered_at) VALUES
    ('ORD-20240801-001', 2, 'DELIVERED',  69.70,  '上海市浦东新区示例路1号', '2026-08-03 10:00:00', '2026-08-03 15:00:00'), -- 4天前, 正常退货/部分退货
    ('ORD-20240805-002', 2, 'SHIPPED',    228.90, '上海市浦东新区示例路1号', '2026-08-05 10:00:00', NULL),                  -- 2天前, 多商品/仅退款被拒
    ('ORD-20240806-003', 2, 'PAID',       89.85,  '上海市浦东新区示例路1号', '2026-08-06 10:00:00', NULL),                  -- 1天前, 仅退款成功+不可退商品
    ('ORD-20240720-004', 3, 'CANCELLED',  150.00, '北京市朝阳区示例路2号',   '2026-07-20 10:00:00', NULL),                  -- 18天前, 订单已取消
    ('ORD-20240725-005', 2, 'DELIVERED',  88.00,  '上海市浦东新区示例路1号', '2026-07-25 10:00:00', '2026-07-25 16:00:00'); -- 13天前, 超7天退货期

-- 商品明细（8 种商品，SKU-006 为定制商品不可退）
INSERT INTO order_items (order_id, item_id, name, price, quantity, returnable) VALUES
    (1, 'SKU-001', '手机壳',         29.90,  1, 1),
    (1, 'SKU-002', '钢化膜',         19.90,  2, 1),
    (2, 'SKU-003', '蓝牙耳机',       199.00, 1, 1),
    (2, 'SKU-004', '耳机收纳盒',     29.90,  1, 1),
    (3, 'SKU-005', '数据线',         29.95,  2, 1),
    (3, 'SKU-006', '定制手机支架',   29.95,  1, 0),
    (4, 'SKU-007', '充电宝',         150.00, 1, 1),
    (5, 'SKU-008', '台灯',           88.00,  1, 1);
