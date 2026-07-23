"""
Tests cho provider_factory.get_provider() — xac nhan co the swap OpenAI/Gemini/Local
chi bang cach doi provider_name/env, khong can sua code o noi goi.
Cac test doi hoi SDK cloud (openai, google-generativeai) hoac llama-cpp-python
se tu skip neu package chua duoc cai trong moi truong hien tai.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.provider_factory import get_provider


def test_get_provider_openai(monkeypatch):
    pytest.importorskip("openai")
    from src.core.openai_provider import OpenAIProvider

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    provider = get_provider(provider_name="openai", model_name="gpt-4o")

    assert isinstance(provider, OpenAIProvider)
    assert provider.model_name == "gpt-4o"


def test_get_provider_google(monkeypatch):
    pytest.importorskip("google.generativeai")
    from src.core.gemini_provider import GeminiProvider

    monkeypatch.setenv("GEMINI_API_KEY", "test-dummy-key")
    provider = get_provider(provider_name="google", model_name="gemini-1.5-flash")

    assert isinstance(provider, GeminiProvider)
    assert provider.model_name == "gemini-1.5-flash"


def test_get_provider_local_missing_model_raises(monkeypatch):
    pytest.importorskip("llama_cpp")

    monkeypatch.setenv("LOCAL_MODEL_PATH", "./models/does-not-exist.gguf")
    with pytest.raises(FileNotFoundError):
        get_provider(provider_name="local")


def test_get_provider_reads_env_default_provider(monkeypatch):
    pytest.importorskip("openai")
    from src.core.openai_provider import OpenAIProvider

    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")

    provider = get_provider()  # no provider_name -> phai doc DEFAULT_PROVIDER

    assert isinstance(provider, OpenAIProvider)


def test_get_provider_invalid_name_raises():
    with pytest.raises(ValueError):
        get_provider(provider_name="not-a-real-provider")
