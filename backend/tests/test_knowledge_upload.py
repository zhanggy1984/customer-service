"""知识库文件上传端点测试（直接调 handler，mock kb_store）。

覆盖：
- 正常上传：文件名 stem 作 source，复用 upsert，返回 {source, count}
- title 覆盖文件名 stem
- 同 source 再传 → upsert 覆盖（幂等语义由 kb_store content_hash 保证）
- 校验：扩展名 415 / 超大小 413 / 空内容 400 / 非 UTF-8 400 / source 截断 100
"""
import io
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, UploadFile

from app.api import routes


def _file(name: str, data: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(data), filename=name)


@pytest.fixture
def mock_kb(monkeypatch):
    upsert = AsyncMock(return_value=3)
    reconcile = AsyncMock(return_value=0)
    monkeypatch.setattr(routes.kb_store, "upsert", upsert)
    monkeypatch.setattr(routes.kb_store, "reconcile_pending", reconcile)
    return upsert, reconcile


@pytest.mark.asyncio
async def test_upload_md_uses_filename_stem_as_source(mock_kb):
    """正常上传 .md：source 取文件名 stem，落库走 kb_store.upsert。"""
    upsert, reconcile = mock_kb
    resp = await routes.upload_knowledge_file(
        file=_file("售后政策.md", "# 售后政策\n七天无理由退货。".encode("utf-8")),
        title=None,
        admin={"username": "admin1"},
    )
    upsert.assert_awaited_once_with("售后政策", "# 售后政策\n七天无理由退货。", "admin1")
    reconcile.assert_awaited_once()
    assert resp == {"source": "售后政策", "count": 3}


@pytest.mark.asyncio
async def test_upload_title_overrides_filename(mock_kb):
    """表单 title 显式传时优先作 source，忽略文件名。"""
    upsert, _ = mock_kb
    await routes.upload_knowledge_file(
        file=_file("whatever.md", "内容".encode("utf-8")),
        title="显式标题",
        admin={"username": "a"},
    )
    upsert.assert_awaited_once_with("显式标题", "内容", "a")


@pytest.mark.asyncio
async def test_upload_same_source_overwrites(mock_kb):
    """同 source 再传 → upsert 再次调用即覆盖；幂等跳检由 kb_store 保证。"""
    upsert, _ = mock_kb
    for _ in range(2):
        await routes.upload_knowledge_file(
            file=_file("faq.md", "内容 v1".encode("utf-8")), title=None, admin={"username": "a"}
        )
    assert upsert.await_count == 2


@pytest.mark.asyncio
async def test_upload_unsupported_extension_415(mock_kb):
    """非 .md/.txt（如 pdf）→ 415（需解析器，不支持）。"""
    with pytest.raises(HTTPException) as exc:
        await routes.upload_knowledge_file(
            file=_file("report.pdf", b"%PDF"), title=None, admin={"username": "a"}
        )
    assert exc.value.status_code == 415


@pytest.mark.asyncio
async def test_upload_empty_content_400(mock_kb):
    """内容全空白 → 400。"""
    with pytest.raises(HTTPException) as exc:
        await routes.upload_knowledge_file(
            file=_file("empty.md", b"   \n  "), title=None, admin={"username": "a"}
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_non_utf8_400(mock_kb):
    """非 UTF-8 编码 → 400。"""
    with pytest.raises(HTTPException) as exc:
        await routes.upload_knowledge_file(
            file=_file("bad.md", b"\xff\xfe\x00\x12"), title=None, admin={"username": "a"}
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_upload_oversize_413(mock_kb):
    """超过 1MB → 413。"""
    with pytest.raises(HTTPException) as exc:
        await routes.upload_knowledge_file(
            file=_file("big.md", b"x" * (routes.KB_UPLOAD_MAX_BYTES + 1)),
            title=None,
            admin={"username": "a"},
        )
    assert exc.value.status_code == 413


@pytest.mark.asyncio
async def test_read_limited_rejects_early_not_full_read():
    """挑战2修复：流式分块读，累计超限即断——总读取量≈上限+1 块，远小于文件全量。"""
    reads = []

    class FakeFile:
        def __init__(self, total):
            self.total = total
            self.pos = 0

        async def read(self, n):
            reads.append(n)
            data = b"x" * min(n, self.total - self.pos)
            self.pos += len(data)
            return data

    with pytest.raises(HTTPException) as exc:
        await routes._read_limited(
            FakeFile(routes.KB_UPLOAD_MAX_BYTES * 10), routes.KB_UPLOAD_MAX_BYTES
        )
    assert exc.value.status_code == 413
    # 每次固定 64KB 分块；累积超限即抛，未把 10MB 全量读入内存
    assert all(n == routes._KB_UPLOAD_CHUNK_SIZE for n in reads)
    assert sum(reads) < routes.KB_UPLOAD_MAX_BYTES * 2


@pytest.mark.asyncio
async def test_upload_source_truncated_to_100(mock_kb):
    """超长文件名 stem 作 source 时截断到 100（source VARCHAR(100)）。"""
    upsert, _ = mock_kb
    await routes.upload_knowledge_file(
        file=_file("长" * 200 + ".md", "内容".encode("utf-8")), title=None, admin={"username": "a"}
    )
    source = upsert.await_args.args[0]
    assert len(source) == 100
