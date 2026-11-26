"""变更四：意图客户端超时与有限重试。

覆盖：
- create_async_client 透传 timeout / max_retries，缺省不改变既有行为
- OrchestratorService.create 仅给 intent_client 读 timeout_seconds / max_retries
  （summary、input_guard 客户端不继承），配置缺省时用 10 / 0
"""
from unittest.mock import MagicMock, patch

import pytest

import app.services.orchestrator_service as orch_module
from app.intent.llm_client import create_async_client
from app.services.orchestrator_service import OrchestratorService


def test_create_async_client_passes_timeout_and_retries():
    with patch("app.intent.llm_client.AsyncOpenAI") as mock_cls:
        create_async_client(
            {"base_url": "http://x", "api_key": "k"},
            timeout=10.0, max_retries=0,
        )
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["timeout"] == 10.0
    assert kwargs["max_retries"] == 0


def test_create_async_client_defaults_omit_overrides():
    """未传时不下发 timeout/max_retries，保持 SDK 默认行为。"""
    with patch("app.intent.llm_client.AsyncOpenAI") as mock_cls:
        create_async_client({"base_url": "http://x", "api_key": "k"})
    kwargs = mock_cls.call_args.kwargs
    assert "timeout" not in kwargs
    assert "max_retries" not in kwargs


def _patch_module(monkeypatch):
    """记录每次客户端创建的 (model_config, timeout, max_retries)。"""
    calls = []

    def fake_create(model_config, timeout=None, max_retries=None):
        calls.append((dict(model_config), timeout, max_retries))
        return MagicMock()

    monkeypatch.setattr(orch_module, "create_async_client", fake_create)
    monkeypatch.setattr(orch_module, "load_intent_config", lambda path: {})
    return calls


@pytest.mark.asyncio
async def test_create_intent_client_reads_client_options_only(monkeypatch):
    """intent_client 收到 10/0；summary 与 input_guard 客户端不继承。"""
    calls = _patch_module(monkeypatch)
    model_config = {
        "models": {
            "default": {"base_url": "http://d", "api_key": "k"},
            "intent_recognizer": {
                "base_url": "http://i", "api_key": "k",
                "timeout_seconds": 10, "max_retries": 0,
            },
            "input_guard": {"base_url": "http://g", "api_key": "k"},
        },
        "prompts": {},
    }

    await OrchestratorService.create(model_config, MagicMock())

    # 三次创建顺序：intent / summary / input_guard
    assert len(calls) == 3
    assert calls[0][1] == 10 and calls[0][2] == 0
    assert calls[1][1] is None and calls[1][2] is None
    assert calls[2][1] is None and calls[2][2] is None


@pytest.mark.asyncio
async def test_create_intent_client_defaults_without_config(monkeypatch):
    """intent_recognizer 段缺省时回退 default，但客户端选项仍用默认 10/0。"""
    calls = _patch_module(monkeypatch)
    model_config = {
        "models": {"default": {"base_url": "http://d", "api_key": "k"}},
        "prompts": {},
    }

    await OrchestratorService.create(model_config, MagicMock())

    assert calls[0][1] == 10 and calls[0][2] == 0
    assert calls[1][1] is None and calls[1][2] is None
