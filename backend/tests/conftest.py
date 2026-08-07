"""pytest 配置：集成测试依赖运行中的服务，不可用时跳过。"""
import httpx
import pytest


def _service_ready() -> bool:
    try:
        resp = httpx.get("http://localhost:8000/healthz", timeout=2)
        return resp.status_code == 200
    except Exception:
        return False


SERVICE_READY = _service_ready()


@pytest.fixture
def service_ready() -> bool:
    return SERVICE_READY
