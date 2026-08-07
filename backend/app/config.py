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
    vector_store: str = "chroma"         # chroma | milvus

    # ---------- 基础设施 ----------
    redis_url: str = "redis://redis:6379/0"
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

    # ---------- RAG ----------
    chroma_persist_dir: str = "./data/chroma"
    # 空 → 嵌入式 PersistentClient（开发单机）；非空 → HttpClient 连接独立 chroma 服务
    chroma_host: str = ""
    chroma_port: int = 8000
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dim: int = 512
    rag_cache_ttl: int = 600
    intent_cache_ttl: int = 60

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
