"""JSON 结构化日志。

用法:
    from app.utils.logger import logger
    logger.info("request_in", extra={"session_id": "s1", "user_id": "u1", "input_len": 42})

extra 中的字段会原样合并进 JSON 输出。Docker 内日志走 stdout（由 docker logging
driver 收集并轮转）；设置环境变量 LOG_TO_FILE=1 时额外落盘 RotatingFileHandler。
"""
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

# logging.LogRecord 内置属性，不可作为业务字段注入
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created))
                 + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # 合并 extra 注入的业务字段（覆盖同名内置字段，业务字段优先）
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                data[k] = v
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, ensure_ascii=False, default=str)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("cs")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = JsonFormatter()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if os.environ.get("LOG_TO_FILE", "0") == "1":
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(log_dir, "app.log"),
            maxBytes=50 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


logger = _build_logger()
