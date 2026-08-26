"""应用配置。从环境变量 / .env 加载（pydantic-settings）。

字段名小写下划线，环境变量名自动映射为大写（deepseek_api_keys -> DEEPSEEK_API_KEYS）。
优先级: 进程环境变量 > .env 文件。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # 本地开发 backend/ 目录下运行时，项目根 .env 在上级目录
        env_file=[".env", "../.env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- DeepSeek ----------
    deepseek_api_keys: str = ""          # 多个 Key 逗号分隔
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model_chat: str = "deepseek-chat"
    deepseek_model_reasoner: str = "deepseek-reasoner"
    deepseek_per_key_rpm: int = 200
    deepseek_queue_max_size: int = 500
    deepseek_queue_timeout: float = 2.0
    deepseek_timeout_chat: float = 8.0
    deepseek_timeout_reasoner: float = 15.0

    # ---------- 运行模式 ----------
    service_mode: str = "local"          # local | remote（对接层）

    # ---------- 基础设施 ----------
    redis_url: str = "redis://redis:6379/1"   # 共享 Redis db index 1（隔离规范 cs=/1）
    mysql_url: str = "mysql+asyncmy://csuser:cspass@mysql:3306/customer_service"
    mysql_pool_size: int = 20
    mysql_max_overflow: int = 40

    # ---------- 认证 ----------
    jwt_secret_key: str = "change-me"
    jwt_expire_hours: int = 2

    # ---------- 会话 ----------
    session_ttl: int = 3600
    conversation_max_rounds: int = 10
    # 会话消息体保存上限（条数）：超出截断，仅保留首条 user 消息 + 最近 N-1 条
    session_max_messages: int = 40
    # 数据保留：会话/tool_call_log 超期清理（回收 MySQL 存储）。判据全部在 MySQL 侧
    # NOW() 与 created_at 的 CURRENT_TIMESTAMP 同基准，Python 不生成 cutoff（避免时区错位）。
    session_retention_days: int = 30               # 保留天数，超期清理
    session_cleanup_interval_seconds: int = 3600   # 定时清理周期
    session_cleanup_batch_size: int = 500          # 单批删除行数，控制事务大小

    # ---------- Agent 工具决策循环（P3） ----------
    # LLM 工具决策循环最大轮数（每轮一次 LLM 调用 + 若干工具执行）
    agent_loop_max_rounds: int = 3
    # 回退闸门：置 True 时 POLICY_INQUIRY 意图决策循环强制补一次 search_policy
    # （纯 LLM 自主导致评测扣分时一键回退，默认关闭）
    agent_loop_force_policy_search: bool = False

    # ---------- RAG ----------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    # 交叉编码重排模型（bge-reranker 真重排，约 1GB，懒加载）
    rerank_model: str = "BAAI/bge-reranker-base"
    rag_cache_ttl: int = 600
    intent_cache_ttl: int = 60
    # 回合级 LLM 答案缓存（P6）：无状态政策轮次整轮缓存，命中零 LLM 调用。
    # KB 变更时即时清（retriever.clear_cache 同源失效），TTL 仅兜底防陈旧。
    turn_cache_enabled: bool = True
    turn_cache_ttl: int = 7200

    # ---------- Milvus（VECTOR_STORE=milvus 时生效） ----------
    milvus_uri: str = "http://milvus:19530"          # compose 内服务名；本地开发改 http://localhost:19533
    milvus_collection: str = "cs_knowledge"   # 共享 Milvus 加 cs_ 前缀（隔离规范）
    milvus_token: str = ""

    # ---------- Admin 默认账号 ----------
    admin_default_username: str = "admin"
    admin_default_password: str = "admin123"

    @property
    def deepseek_api_key_list(self) -> list[str]:
        return [k.strip() for k in self.deepseek_api_keys.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
