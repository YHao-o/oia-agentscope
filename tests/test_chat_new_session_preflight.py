"""变更二：新建无附件会话跳过空历史/空快照。

覆盖 routes 透传的 `is_new_session` + chat_service 的
`skip_empty_preflight = is_new_session and not _contains_data_blocks(...)`：
- 新建纯文本：不调 load_messages；跳过轮前空快照（list_session_files 仅轮末一次）
- 新建含 data 块：保留既有路径（load_messages 调用、快照两次）
- 已有会话：历史与快照正常加载
- is_new_session 缺省（False）：行为与改动前一致
"""
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.chat_service as chat_service_module
from app.services.chat_service import _contains_data_blocks, generate_response

_SAFE_SENS = {
    "blocked": False, "reason": "", "hit_sources": [],
    "sensitive_words": [], "source": "server", "raw": {},
}


def _patch_sensitive(monkeypatch):
    async def fake_dict(text, stage="input"):
        return dict(_SAFE_SENS)

    async def fake_check(text, stage="output"):
        return dict(_SAFE_SENS)

    monkeypatch.setattr(chat_service_module, "dict_check_sensitive", fake_dict)
    monkeypatch.setattr(chat_service_module, "check_sensitive", fake_check)


def _patch_recommended_questions(monkeypatch):
    async def fake_emit_rq(*args, **kwargs):
        yield 'data: {"type": "recommended_questions", "questions": []}\n\n'

    monkeypatch.setattr(chat_service_module, "_emit_recommended_questions", fake_emit_rq)


def _make_orch(captured):
    orch = MagicMock()
    # 预检围栏客户端置空：不触发 403 预检外呼
    orch._input_probe_client = None
    orch._input_probe_model_cfg = None

    async def fake_run(messages, **kwargs):
        captured["messages"] = messages
        yield 'data: {"type": "summary", "content": "回答"}\n\n'

    orch.run = fake_run
    return orch


def _make_session_service(saved=None):
    svc = MagicMock()
    svc.load_messages = AsyncMock(return_value=list(saved or []))
    svc.append_messages = AsyncMock(return_value={
        "user_message_id": 101, "assistant_message_id": 102,
    })
    svc.save_latest_trace_id = AsyncMock()
    svc.append_session_files = AsyncMock()
    svc.mark_last_assistant_failed = AsyncMock()
    svc.save_agent_state = AsyncMock()
    return svc


def _make_workspace_manager():
    wm = MagicMock()
    wm.list_session_files = AsyncMock(return_value={})
    return wm


def _make_request():
    request = MagicMock()
    request.app.state.upload_file_dao = None
    request.app.state.redis_client = None
    return request


async def _generate(monkeypatch, messages, *, session_service, wm, is_new_session=None):
    _patch_sensitive(monkeypatch)
    _patch_recommended_questions(monkeypatch)
    captured = {}
    orch = _make_orch(captured)
    kwargs = dict(
        orchestrator_service=orch,
        messages=messages,
        session_id="sess-1",
        user_id="u1",
        session_service=session_service,
        langfuse_service=None,
        request=_make_request(),
        workspace_manager=wm,
    )
    if is_new_session is not None:
        kwargs["is_new_session"] = is_new_session
    raw = [ev async for ev in generate_response(**kwargs)]
    return captured, raw


_SAVED_HISTORY = [{"role": "user", "content": "旧问题", "agent_ids": []}]


def test_contains_data_blocks_detection():
    assert _contains_data_blocks([]) is False
    assert _contains_data_blocks([{"role": "user", "content": "纯文本"}]) is False
    assert _contains_data_blocks([
        {"role": "user", "content": [{"type": "text", "text": "x"}]},
    ]) is False
    assert _contains_data_blocks([
        {"role": "user", "content": [{"type": "data", "data": "x"}]},
    ]) is True


@pytest.mark.asyncio
async def test_new_session_text_skips_history_and_pre_snapshot(monkeypatch):
    """新建纯文本：不调 load_messages，轮前快照被跳过（仅轮末 after 快照）。"""
    messages = [{"role": "user", "content": "你好"}]
    session_service = _make_session_service(_SAVED_HISTORY)
    wm = _make_workspace_manager()

    captured, _ = await _generate(
        monkeypatch, messages, session_service=session_service, wm=wm,
        is_new_session=True,
    )

    session_service.load_messages.assert_not_awaited()
    # 轮前 before 快照跳过 → list_session_files 仅轮末 after 快照一次
    assert wm.list_session_files.await_count == 1
    # 模型收到且仅收到当前消息（不含历史）
    assert captured["messages"] == [{"role": "user", "content": "你好"}]


@pytest.mark.asyncio
async def test_new_session_with_data_block_keeps_preflight(monkeypatch):
    """新建含 data 块：保留既有路径（历史加载 + 轮前/轮末两次快照）。"""
    messages = [{
        "role": "user",
        "content": [{"type": "data", "data": "x"}, {"type": "text", "text": "看图"}],
    }]
    session_service = _make_session_service(_SAVED_HISTORY)
    wm = _make_workspace_manager()

    captured, _ = await _generate(
        monkeypatch, messages, session_service=session_service, wm=wm,
        is_new_session=True,
    )

    session_service.load_messages.assert_awaited_once()
    assert wm.list_session_files.await_count == 2
    # 历史被合并进 full_messages
    assert captured["messages"][0]["content"] == "旧问题"
    assert captured["messages"][-1] == messages[0]


@pytest.mark.asyncio
async def test_existing_session_loads_history_and_snapshot(monkeypatch):
    """已有会话：历史与快照正常加载。"""
    messages = [{"role": "user", "content": "继续"}]
    session_service = _make_session_service(_SAVED_HISTORY)
    wm = _make_workspace_manager()

    captured, _ = await _generate(
        monkeypatch, messages, session_service=session_service, wm=wm,
        is_new_session=False,
    )

    session_service.load_messages.assert_awaited_once()
    assert wm.list_session_files.await_count == 2
    assert captured["messages"][0]["content"] == "旧问题"


@pytest.mark.asyncio
async def test_default_flag_preserves_existing_behavior(monkeypatch):
    """不传 is_new_session 时默认 False，行为与改动前一致。"""
    messages = [{"role": "user", "content": "继续"}]
    session_service = _make_session_service(_SAVED_HISTORY)
    wm = _make_workspace_manager()

    captured, _ = await _generate(
        monkeypatch, messages, session_service=session_service, wm=wm,
    )

    session_service.load_messages.assert_awaited_once()
    assert wm.list_session_files.await_count == 2
    assert captured["messages"][0]["content"] == "旧问题"
