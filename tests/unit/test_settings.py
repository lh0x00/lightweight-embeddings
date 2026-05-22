"""Settings loading + helpers."""

from __future__ import annotations

from lightweight_embeddings.settings import Settings, reset_settings


def test_models_preload_list_default():
    s = Settings()
    assert s.models_preload_list == ["multilingual-e5-small"]


def test_models_preload_list_csv(monkeypatch):
    reset_settings()
    monkeypatch.setenv("LWE_MODELS_PRELOAD", "multilingual-e5-small, bge-m3")
    s = Settings()
    assert s.models_preload_list == ["multilingual-e5-small", "bge-m3"]


def test_models_preload_none(monkeypatch):
    reset_settings()
    monkeypatch.setenv("LWE_MODELS_PRELOAD", "none")
    s = Settings()
    assert s.models_preload_list == []


def test_models_preload_star(monkeypatch):
    reset_settings()
    monkeypatch.setenv("LWE_MODELS_PRELOAD", "*")
    s = Settings()
    assert s.models_preload_list == ["*"]


def test_cors_origins_csv(monkeypatch):
    reset_settings()
    monkeypatch.setenv("LWE_CORS_ORIGINS", "https://a.example, https://b.example")
    s = Settings()
    assert s.cors_origins == ["https://a.example", "https://b.example"]
