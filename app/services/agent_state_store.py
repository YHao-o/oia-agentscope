"""AgentState 存取模块：加载（含脏数据迁移/截断）、保存与历史枚举修正。

从 orchestrator_service.py 外移的纯函数，不依赖服务实例状态；
调用方传入 session_service 与会话参数即可。
"""
import logging
from typing import Any, Optional

from agentscope.state import AgentState

from app.utils.event_serialization import sanitize_text_recursive

logger = logging.getLogger(__name__)

# 历史上下文保留条数（6 轮 = 12 条 user/assistant 消息），用于截断 AgentState.context
_HISTORY_KEEP_LAST = 12


def migrate_legacy_enum_values(state_dict: dict) -> dict:
    """修正历史脏数据中枚举被 str() 序列化的问题。

    根因：早期保存使用 model_dump() + json.dumps(default=str)，枚举实例
    （如 PermissionMode.BYPASS）被 str() 转成 "PermissionMode.BYPASS"
    而非 "bypass"，导致反序列化时 PermissionMode("PermissionMode.BYPASS")
    抛 ValueError。此处检测并修正为合法的枚举值。
    """
    if not isinstance(state_dict, dict):
        return state_dict

    # permission_context.mode: "PermissionMode.BYPASS" → "bypass"
    perm = state_dict.get("permission_context")
    if isinstance(perm, dict):
        mode = perm.get("mode")
        if isinstance(mode, str) and mode.startswith("PermissionMode."):
            perm["mode"] = mode.split(".", 1)[1].lower()
    return state_dict


def trim_state_context(state_dict: dict, keep_last: int = _HISTORY_KEEP_LAST) -> dict:
    """截断 AgentState.context 为最近 keep_last 条（默认 6 = 3 轮）。

    AgentState.context 是完整对话历史 Msg_dict 列表；大模型上下文有限时
    仅保留最近 keep_last 条（默认 6 = 3 轮）。仅内存截断，不落库。
    """
    if not state_dict:
        return state_dict
    ctx = state_dict.get("context")
    if isinstance(ctx, list) and len(ctx) > keep_last:
        state_dict = {**state_dict, "context": ctx[-keep_last:]}
    return state_dict


async def load_agent_state(
    session_service: Any,
    session_id: Optional[str],
    agent_id: str,
) -> Optional[AgentState]:
    """加载单个 agent 的 AgentState：从 session_service 读取 + 历史脏数据迁移
    + trim 截断 + 异常兜底。

    跨智能体上下文共享：当目标 agent 无历史状态时，回退到同会话下最近
    更新的 agent 状态，继承其对话历史（context + summary），但重置
    agent 特定字段（cur_iter、reply_id、tool_context 等），保证
    用户在同一会话内切换智能体时能看到之前的对话内容。

    session_service 为空或读取失败均返回 None（调用方创建新状态）。
    反序列化失败时记录 warning（含异常栈），便于定位 schema 不兼容问题。
    """
    if not (session_service and session_id):
        return None
    try:
        state_dict = await session_service.load_agent_state(session_id, agent_id)
    except Exception:
        logger.warning(
            f"[OrchestratorService] 读取 {agent_id} 状态失败，将新建",
            exc_info=True,
        )
        return None

    # 目标 agent 无历史状态：尝试继承同会话最近 agent 的对话上下文
    if not state_dict:
        try:
            latest = await session_service.get_latest_agent_state(session_id)
        except Exception:
            logger.warning(
                "[agent_state_store] 查询最近 agent 状态失败，将新建 "
                "session=%s agent=%s", session_id, agent_id, exc_info=True,
            )
            latest = None

        if not latest:
            logger.info(
                "[agent_state_store] 无历史状态 session=%s agent=%s",
                session_id, agent_id,
            )
            return None

        # 防御性校验：latest 应为 dict；若非 dict（历史数据损坏），
        # 退化为新建状态，避免 .get() 抛 AttributeError。
        if not isinstance(latest, dict):
            logger.warning(
                "[agent_state_store] 最近 agent 状态非 dict，跳过继承 "
                "session=%s agent=%s type=%s",
                session_id, agent_id, type(latest).__name__,
            )
            return None

        # 继承对话历史（context + summary），重置 agent 特定字段：
        # - cur_iter 重置为 0（新 agent 从第 0 轮开始）
        # - reply_id 重新生成
        # - tool_context 使用默认（不同 agent 工具集不同，避免继承旧激活状态）
        inherited_ctx = latest.get("context") or []
        if not isinstance(inherited_ctx, list):
            inherited_ctx = []
        inherited_summary = latest.get("summary") or ""
        state_dict = {
            "session_id": session_id,
            "context": inherited_ctx,
            "summary": inherited_summary,
            "permission_context": latest.get("permission_context")
            or {"mode": "bypass"},
        }
        logger.info(
            "[agent_state_store] 目标 agent 无状态，继承最近对话上下文 "
            "session=%s agent=%s inherited_context_len=%d",
            session_id, agent_id, len(inherited_ctx),
        )
    else:
        ctx_len = len(state_dict.get("context") or [])
        logger.info(
            "[agent_state_store] 加载状态 session=%s agent=%s context_len=%d",
            session_id, agent_id, ctx_len,
        )

    # 历史脏数据迁移：修正 "PermissionMode.BYPASS" → "bypass" 等
    state_dict = migrate_legacy_enum_values(state_dict)
    # 截断 context 为最近 N 条，控制模型输入 token
    state_dict = trim_state_context(state_dict)
    try:
        return AgentState.model_validate(state_dict)
    except Exception:
        logger.warning(
            f"[OrchestratorService] 反序列化 {agent_id} 状态失败，将新建",
            exc_info=True,
        )
        return None


def _history_to_msgs(history: Optional[list]) -> list:
    """把会话级历史（[{"role","content"}]）转成 AgentState 可用的 Msg 列表。

    只保留 role 为 user/assistant 且 content 非空的条目，其余（工具轨迹、
    system、空内容）一律跳过——它们不属于"聊过什么"的会话记忆。
    """
    from agentscope.message import AssistantMsg, TextBlock, UserMsg

    msgs: list = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        blocks = [TextBlock(text=str(content))]
        if role == "user":
            msgs.append(UserMsg(name="user", content=blocks))
        else:
            msgs.append(AssistantMsg(name="assistant", content=blocks))
    return msgs


def inject_session_context(
    agent_state: Optional[AgentState],
    session_id: Optional[str],
    history: Optional[list],
) -> Optional[AgentState]:
    """把会话级历史注入 agent 上下文，保证跨智能体共享整场对话记忆。

    多轮记忆原本按 agent 隔离（agent_states 表以 (session_id, agent_id) 为
    主键），导致用户在同一会话内切换智能体、或切回默认问答时，作答模型只能
    看到该 agent 自己的（可能过期/被截断的）上下文。这里在每次执行前用会话级
    历史覆盖 agent.context——这份历史与查询改写/意图识别用的是同一个窗口，
    因此"聊过什么"成为会话级资产，不再随智能体切换而丢失。

    - history 为空（或全为无效条目）时原样返回；
    - agent_state 为 None（该 agent 在本会话从未执行过）时新建一个带会话
      历史的 AgentState，权限沿用 bypass（与 registry 新建 agent 时一致）；
    - 只替换 context，保留 summary 及其它 agent 特定字段。
    """
    try:
        msgs = _history_to_msgs(history)
    except Exception:
        logger.warning(
            "[agent_state_store] 会话历史转换失败，跳过注入 session=%s",
            session_id, exc_info=True,
        )
        return agent_state
    if not msgs:
        return agent_state

    is_new = agent_state is None
    if is_new:
        from agentscope.permission import PermissionContext, PermissionMode

        agent_state = AgentState(
            session_id=session_id or "",
            permission_context=PermissionContext(mode=PermissionMode.BYPASS),
        )
    agent_state.context = msgs
    logger.debug(
        "[agent_state_store] 注入会话历史 session=%s 条数=%d 新建=%s",
        session_id, len(msgs), is_new,
    )
    return agent_state


async def persist_agent_state(
    session_service: Any,
    session_id: Optional[str],
    user_id: Optional[str],
    agent_id: str,
    state_dict: dict,
) -> None:
    """保存单个 agent 的 AgentState（带非空校验与异常兜底）。"""
    if not (session_service and session_id and user_id and state_dict):
        return
    # 全链路清洗：state.context 含模型原始输出 + 工具结果，
    # 流式 chunk 切断的 emoji（孤立代理）和 bash 输出的控制字符
    # （null/ANSI/二进制片段）都会导致 DAO 层（MySQL/Redis）
    # UTF-8 编码失败、state 丢失。升级为 sanitize_text_recursive，
    # 两类问题一起兜住
    state_dict = sanitize_text_recursive(state_dict)
    try:
        await session_service.save_agent_state(
            session_id, user_id, agent_id, state_dict,
        )
    except Exception:
        logger.exception(
            f"[OrchestratorService] 保存 agent {agent_id} 状态失败"
        )
