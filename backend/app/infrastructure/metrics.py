"""轻量 Prometheus 文本指标（零依赖，进程内存）。

覆盖生产排障最关心的信号：LLM 调用量/延迟/熔断/排队、会话锁等待、意图规则命中。
GET /metrics 直接 render() 输出 Prometheus 文本格式（见 main.py）。

多进程注意：指标存进程内存，uvicorn 多 worker 下各自独立。当前 docker 单 worker
部署无此问题；若将来扩 worker 需换 prometheus_client 并接入聚合网关。
"""
import time
from collections import defaultdict

# key = (metric_name, frozenset(label_items))；frozenset 保证标签无序可哈希
_counters: dict[tuple[str, frozenset], int] = defaultdict(int)
_sum: dict[tuple[str, frozenset], float] = defaultdict(float)
_count: dict[tuple[str, frozenset], int] = defaultdict(int)
_gauges: dict[tuple[str, frozenset], float] = defaultdict(float)


def _key(name: str, labels: dict | None) -> tuple[str, frozenset]:
    return (name, frozenset((labels or {}).items()))


def inc(name: str, labels: dict | None = None, value: int = 1) -> None:
    """计数器累加（Prometheus 输出带 *_total 后缀，命名时去掉 total）。"""
    _counters[_key(name, labels)] += value


def observe(name: str, value: float, labels: dict | None = None) -> None:
    """summary 观测（输出 {name}_sum / {name}_count，可算均值/总量）。"""
    k = _key(name, labels)
    _sum[k] += value
    _count[k] += 1


def set_gauge(name: str, value: float, labels: dict | None = None) -> None:
    """gauge 设置（当前值快照，如队列深度）。"""
    _gauges[_key(name, labels)] = value


def _render_labels(label_items: frozenset) -> str:
    if not label_items:
        return ""
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(label_items))
    return "{" + inner + "}"


def render() -> str:
    """Prometheus 文本格式输出。名称排序保证输出稳定，便于 diff/断言。"""
    lines: list[str] = []
    for (name, labels), v in sorted(_counters.items()):
        lines.append(f"{name}_total{_render_labels(labels)} {v}")
    for (name, labels), v in sorted(_count.items()):
        lines.append(f"{name}_count{_render_labels(labels)} {v}")
    for (name, labels), v in sorted(_sum.items()):
        lines.append(f"{name}_sum{_render_labels(labels)} {v}")
    for (name, labels), v in sorted(_gauges.items()):
        lines.append(f"{name}{_render_labels(labels)} {v}")
    return "\n".join(lines) + "\n" if lines else ""
