"""Lightweight Embeddings — package entry point.

Importing this module is **side-effect free**: model weights are loaded
lazily inside :func:`main.create_app`'s lifespan. The previous version eager
loaded ten models on import, which made tests, CLI tools, and worker boot
prohibitively expensive.
"""

from __future__ import annotations

__version__ = "1.1.0"
__author__ = "lamhieu"
__description__ = "Fast, lightweight, multilingual embeddings & reranking service."


def _build_default_app():
    """Build the application using settings from the environment.

    Re-exposed as the module-level ``app`` for backwards compatibility with
    the existing entry script ``app.py``::

        from lightweight_embeddings import app
    """
    from .main import create_app

    return create_app()


# Lazy module attribute so importing the package alone does not boot FastAPI
# or load any model. The default ASGI entry expression ``app:app`` (where
# ``app.py`` does ``from lightweight_embeddings import app``) still works
# because Python evaluates the attribute on demand.
def __getattr__(name: str):  # pragma: no cover - simple delegation
    if name == "app":
        global app  # type: ignore[name-defined]
        app = _build_default_app()
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["__author__", "__description__", "__version__", "app"]
