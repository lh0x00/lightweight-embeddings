"""Centralized logging configuration.

Supports two formats:

* Plain text (default) — readable for local development.
* JSON (``LWE_LOG_JSON=true``) — structured logs for production.

A small filter drops uvicorn access logs for ``/v1`` routes (configurable)
since those are recorded explicitly by application metrics.
"""

from __future__ import annotations

import json
import logging
import logging.config
import sys
from typing import Any, ClassVar

from .settings import Settings


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter without external dependencies."""

    _RESERVED: ClassVar[set[str]] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # extra fields attached via logger.info(..., extra={...})
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


class AccessLogPathFilter(logging.Filter):
    """Filter uvicorn access log records by URL path substring.

    Returns ``False`` (drop) for records whose request line / formatted
    message contains any of the configured prefixes. We scan both the
    rendered message *and* the args tuple because uvicorn's access log
    layout varies between versions: in some it is ``args[1]`` (request
    line ``"GET /v1/foo HTTP/1.1"``), in others ``args[2]``.
    """

    def __init__(self, drop_prefixes: tuple[str, ...] = ("/v1",)) -> None:
        super().__init__()
        self.drop_prefixes = drop_prefixes

    def filter(self, record: logging.LogRecord) -> bool:
        candidates: list[str] = []
        if isinstance(record.args, (tuple, list)):
            candidates.extend(str(a) for a in record.args)
        elif record.args:
            candidates.append(str(record.args))
        try:
            candidates.append(record.getMessage())
        except Exception:
            pass
        for text in candidates:
            for prefix in self.drop_prefixes:
                if prefix in text:
                    return False
        return True


def configure_logging(settings: Settings) -> None:
    """Apply logging configuration based on application settings.

    Idempotent: re-applies the configuration on every call.
    """
    formatter: dict[str, Any]
    if settings.log_json:
        formatter = {"()": f"{__name__}.JsonFormatter"}
    else:
        formatter = {
            "format": (
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
            ),
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        }

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": formatter},
        "filters": {
            "drop_v1_access": {
                "()": f"{__name__}.AccessLogPathFilter",
                "drop_prefixes": ("/v1",),
            },
        },
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": "default",
            },
        },
        "loggers": {
            "": {  # root
                "handlers": ["stdout"],
                "level": settings.log_level,
            },
            "uvicorn": {
                "handlers": ["stdout"],
                "level": settings.log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["stdout"],
                "level": settings.log_level,
                "propagate": False,
                "filters": ["drop_v1_access"],
            },
            "uvicorn.error": {
                "handlers": ["stdout"],
                "level": settings.log_level,
                "propagate": False,
            },
            # quiet some chatty libs
            "httpx": {"level": "WARNING"},
            "httpcore": {"level": "WARNING"},
            "PIL": {"level": "WARNING"},
            "transformers": {"level": "WARNING"},
            "sentence_transformers": {"level": "WARNING"},
        },
    }

    logging.config.dictConfig(config)
