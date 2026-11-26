"""变更一：单意图跳过第二次编排 LLM（fast path）。

覆盖 `OrchestratorService._run_with_workspace_task` 的"⑤ 意图编排"段：
- 单意图（len==1）不调用 plan_orchestration，使用 independent/[]/{} 等价默认值
- 多意图保留第二次 LLM 及 execution_order 重排
- 单意图仍下发 intent-orchestration started 与 intents_recognized（前端进度不变）
- 单意图路径下工作区准备失败的错误语义不变
"""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

import app.services.orchestrator_service as orch_module
from app.intent.models import Intent
from app.services.orchestrator_service import OrchestratorService
from app.services.round_state import RoundState


class _FakeOrchestrator:
    """最小编排器：记录 intent_result，run 不产出业务事件。"""

    def __init__(self):
        self.intent_result = None
        self._last_results = []

    async def run(self, intent_result, **kwargs):
        self.intent_result = intent_result
        return
        yield  # pragma: no cover


def _parse(events):
    return [json.loads(ev[6:].strip()) for ev in events if ev.startswith("data: ")]


def _make_service(fake_orch) -> OrchestratorService:
    svc = OrchestratorService.__new__(OrchestratorService)
    svc._orchestrator_params = {
        "parallel_timeout": 5,
        "pipeline_step_timeout": 5,
        "react_max_steps": 3,
    }
    svc._intent_client = object()
    svc._intent_model_cfg = {}
    svc._workspace_manager = MagicMock()
    svc._create_orchestrator = lambda mode, factory: fake_orch
    return svc


def _make_recognizer(intents, plan_ret=("independent", [], {})):
    rec = MagicMock()
    rec.recognize_intents = AsyncMock(return_value=(intents, {}))
    rec.plan_orchestration = AsyncMock(return_value=plan_ret)
    rec.get_orchestration_mode = MagicMock(return_value="pipeline")
    return rec


def _patch_io(monkeypatch):
    """屏蔽上传/工作区/状态存取等外部 IO，聚焦编排决策分支。"""
    async def _no_uploads(request, session_id):
        return False

    async def _no_hint(request, session_id):
        return ""

    async def _no_wait(request, session_id):
        if False:
            yield

    async def _persist(request, session_id, user_id, wm):
        return {"file_ids": [], "sandbox_files": []}

    async def _no_ctx(request, session_id, sandbox_files):
        return ""

    async def _load_state(session_service, session_id, agent_id):
        return None

    def _inject(state, session_id, history):
        return state

    monkeypatch.setattr(orch_module, "has_unbound_uploads", _no_uploads)
    monkeypatch.setattr(orch_module, "build_recognizer_file_hint", _no_hint)
    monkeypatch.setattr(orch_module, "wait_for_upload_parsing", _no_wait)
    monkeypatch.setattr(orch_module, "persist_upload_files_to_workspace", _persist)
    monkeypatch.setattr(orch_module, "load_upload_context", _no_ctx)
    monkeypatch.setattr(orch_module, "load_agent_state", _load_state)
    monkeypatch.setattr(orch_module, "inject_session_context", _inject)


async def _ws_ok():
    return (MagicMock(), MagicMock())


async def _run(svc, rec, ws_coro):
    """跑完 _run_with_workspace_task，返回 (前端事件, round_state)。"""
    rewriter = MagicMock()
    rewriter.rewrite = AsyncMock(return_value=("改写后", {}))
    ws_task = asyncio.create_task(ws_coro)
    rs = RoundState()
    events = [
        ev async for ev in svc._run_with_workspace_task(
            ws_task=ws_task,
            rewriter=rewriter,
            recognizer=rec,
            user_input="你好",
            history=[],
            messages=[{"role": "user", "content": "你好"}],
            session_id="sess-1",
            user_id="u1",
            session_service=MagicMock(),
            agent_id=None,
            langfuse_service=None,
            round_state=rs,
        )
    ]
    return events, rs


@pytest.mark.asyncio
async def test_single_intent_skips_plan_orchestration(monkeypatch):
    """单意图：plan_orchestration 调用 0 次，mode 仍按 single_intent 配置。"""
    _patch_io(monkeypatch)
    fake = _FakeOrchestrator()
    svc = _make_service(fake)
    rec = _make_recognizer(
        [Intent(id="general_chat", query="你好", agent="general_agent")]
    )

    await _run(svc, rec, _ws_ok())

    assert rec.plan_orchestration.await_count == 0
    assert fake.intent_result is not None
    assert fake.intent_result.relation == "independent"
    assert fake.intent_result.execution_order == []
    # 编排模式仍由 get_orchestration_mode(intent_result) 决定（配置选择 pipeline）
    rec.get_orchestration_mode.assert_called_once_with(fake.intent_result)


@pytest.mark.asyncio
async def test_single_intent_keeps_intent_events_before_execution(monkeypatch):
    """单意图：orchestration started 与 intents_recognized 事件顺序不变。"""
    _patch_io(monkeypatch)
    fake = _FakeOrchestrator()
    svc = _make_service(fake)
    rec = _make_recognizer(
        [Intent(id="general_chat", query="你好", agent="general_agent")]
    )

    events, _ = await _run(svc, rec, _ws_ok())
    parsed = _parse(events)
    types = [e["type"] for e in parsed]

    assert "intents_recognized" in types
    started_idx = next(
        i for i, e in enumerate(parsed)
        if e["type"] == "intent_step" and e.get("phase") == "orchestration"
        and e.get("status") == "started"
    )
    assert "intent_step" in types
    assert started_idx < types.index("intents_recognized")
    assert fake.intent_result is not None


@pytest.mark.asyncio
async def test_multi_intent_calls_plan_orchestration_once(monkeypatch):
    """多意图：plan_orchestration 调用 1 次，relation/order 与重排生效。"""
    _patch_io(monkeypatch)
    fake = _FakeOrchestrator()
    svc = _make_service(fake)
    rec = _make_recognizer(
        [
            Intent(id="a", query="qa", agent="agent_a"),
            Intent(id="b", query="qb", agent="agent_b"),
        ],
        plan_ret=("related_fixed", [1, 0], {}),
    )

    await _run(svc, rec, _ws_ok())

    assert rec.plan_orchestration.await_count == 1
    assert fake.intent_result.relation == "related_fixed"
    assert fake.intent_result.execution_order == [1, 0]
    # execution_order=[1,0] → 原 [a, b] 重排为 [b, a]
    assert [i.id for i in fake.intent_result.intents] == ["b", "a"]


@pytest.mark.asyncio
async def test_single_intent_workspace_failure_semantics_unchanged(monkeypatch):
    """单意图跳过路径下工作区准备失败：错误语义与跳过决策均不变。"""
    _patch_io(monkeypatch)
    fake = _FakeOrchestrator()
    svc = _make_service(fake)
    rec = _make_recognizer(
        [Intent(id="general_chat", query="你好", agent="general_agent")]
    )

    async def _ws_fail():
        raise RuntimeError("sandbox down")

    events, rs = await _run(svc, rec, _ws_fail())
    errors = [e for e in _parse(events) if e["type"] == "error"]

    assert len(errors) == 1
    assert errors[0]["message"].startswith("环境准备失败")
    assert rs.success is False
    assert rec.plan_orchestration.await_count == 0
