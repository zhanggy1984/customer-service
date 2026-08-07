"""对接层异常定义。"""


class ServiceUnavailableException(Exception):
    """数据库/下游服务不可用。

    重试耗尽后抛出。Agent 层捕获后回复"系统出问题了，请稍后重试"。
    """
