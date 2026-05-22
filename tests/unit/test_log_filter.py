"""LogFilter must drop /v1 access logs and keep everything else."""

from __future__ import annotations

import logging

from lightweight_embeddings.logging_config import AccessLogPathFilter


def _make_record(request_line: str) -> logging.LogRecord:
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg='%s - "%s" %s',
        args=("127.0.0.1:80", request_line, "200"),
        exc_info=None,
    )
    return record


def test_filter_drops_v1():
    f = AccessLogPathFilter(("/v1",))
    record = _make_record("POST /v1/embeddings HTTP/1.1")
    assert f.filter(record) is False


def test_filter_keeps_health():
    f = AccessLogPathFilter(("/v1",))
    record = _make_record("GET /healthz HTTP/1.1")
    assert f.filter(record) is True


def test_filter_keeps_short_args():
    f = AccessLogPathFilter(("/v1",))
    record = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="short",
        args=("just-one-arg",),
        exc_info=None,
    )
    assert f.filter(record) is True
