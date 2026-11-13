"""Todo conditioning and census through /v1/chat/completions.

Two guarantees:
  1. A request carrying the ``todo_write`` tool gets the completion reminder in
     its system message and tool description — but only for a model whose
     ``todo_conditioning`` gate is on.
  2. Every todo request records a census row, whether or not conditioning ran,
     so the behavior is measurable before anything is switched over.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import select


class _Usage:
    prompt_tokens = 10
    completion_tokens = 5
    completion_tokens_details = None


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments, index=0):
        self.index = index
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls
        self.content = "done"


class _Choice:
    def __init__(self, finish_reason="stop", tool_calls=None):
        self.finish_reason = finish_reason
        self.message = _Message(tool_calls)


class _Response:
    """A non-streaming completion with the attributes the route reads."""

    def __init__(self, tool_calls=None, finish_reason="stop"):
        self.choices = [_Choice(finish_reason, tool_calls)]
        self.usage = _Usage()
        self.model = "dummy"

    def model_dump(self):
        return {"id": "x", "choices": [], "usage": None}


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StreamChunk:
    """A streaming chunk; the route reads usage, the first choice, and model_dump."""

    def __init__(self, usage=None, delta=None, finish_reason=None):
        self.usage = usage
        self.choices = []
        if delta is not None or finish_reason is not None:
            choice = MagicMock()
            choice.delta = delta
            choice.finish_reason = finish_reason
            self.choices = [choice]

    def model_dump(self):
        return {"choices": []}


_TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "todo_write",
        "description": "Record and update a task list.",
        "parameters": {"type": "object"},
    },
}

_OPEN_TODOS = [
    {"content": "clone the fork", "status": "completed"},
    {"content": "sync the branch", "status": "in_progress"},
]

_FINISHED_TODOS = [
    {"content": "clone the fork", "status": "completed"},
    {"content": "sync the branch", "status": "completed"},
]


def _history(todos):
    return [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "do the work"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "todo_write", "arguments": json.dumps({"todos": todos})},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Updated todo list."},
    ]


@pytest.fixture
def billable_key(app, test_user, test_model_endpoint):
    """API key for test_user with an unlimited coin pool and a healthy endpoint."""
    with app.app_context():
        from lumen.extensions import db
        from lumen.models.api_key import APIKey
        from lumen.models.entity_limit import EntityLimit
        from lumen.services.crypto import hash_api_key
        db.session.add(EntityLimit(entity_id=test_user["id"], max_coins=-2, refresh_coins=0, starting_coins=0))
        db.session.add(APIKey(entity_id=test_user["id"], name="k", key_hash=hash_api_key("lk_todo_tok")))
        db.session.commit()
    return "lk_todo_tok"


def _set_conditioning(app, model_id, enabled):
    with app.app_context():
        from lumen.extensions import db
        from lumen.models.model_config import ModelConfig
        from lumen.services import model_cache
        db.session.get(ModelConfig, model_id).todo_conditioning = enabled
        db.session.commit()
    # The request path reads models from the process-wide snapshot.
    model_cache.clear()


def _mock_openai(response):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = response
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_client), mock_client


def _census(app):
    with app.app_context():
        from lumen.extensions import db
        from lumen.models.request_log import RequestLog
        rows = db.session.execute(select(RequestLog)).scalars().all()
        assert len(rows) == 1, "exactly one request should have been billed"
        return rows[0].todo_census


def test_gate_off_leaves_the_request_alone_but_still_records_a_census(
        app, client, test_user, test_model, billable_key):
    mock_cls, mock_client = _mock_openai(_Response())
    with patch("lumen.blueprints.api.routes.openai.OpenAI", mock_cls):
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer lk_todo_tok"},
            json={"model": "test-model", "messages": _history(_OPEN_TODOS), "tools": [_TODO_TOOL]},
        )
    assert resp.status_code == 200

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][0]["content"] == "You are a coding agent."
    assert kwargs["tools"][0]["function"]["description"] == "Record and update a task list."

    census = _census(app)
    assert census == {
        "items": 2, "completed": 1, "in_progress": 1, "pending": 0,
        "called": False, "all_completed": False, "finish_reason": "stop",
    }


def test_gate_on_appends_the_reminder_to_system_prompt_and_description(
        app, client, test_user, test_model, billable_key):
    _set_conditioning(app, test_model["id"], True)
    mock_cls, mock_client = _mock_openai(
        _Response(tool_calls=[_ToolCall("todo_write", json.dumps({"todos": _FINISHED_TODOS}))],
                  finish_reason="tool_calls"))
    with patch("lumen.blueprints.api.routes.openai.OpenAI", mock_cls):
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer lk_todo_tok"},
            json={"model": "test-model", "messages": _history(_OPEN_TODOS), "tools": [_TODO_TOOL]},
        )
    assert resp.status_code == 200

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    system = kwargs["messages"][0]["content"]
    assert system.startswith("You are a coding agent.")
    assert "[lumen:todo-write]" in system
    assert "[lumen:todo-write]" in kwargs["tools"][0]["function"]["description"]

    # The response's own call is what the census reports.
    census = _census(app)
    assert census["called"] is True and census["all_completed"] is True
    assert census["finish_reason"] == "tool_calls"


def test_conditioning_is_skipped_for_a_request_without_the_tool(
        app, client, test_user, test_model, billable_key):
    _set_conditioning(app, test_model["id"], True)
    mock_cls, mock_client = _mock_openai(_Response())
    with patch("lumen.blueprints.api.routes.openai.OpenAI", mock_cls):
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer lk_todo_tok"},
            json={"model": "test-model",
                  "messages": [{"role": "system", "content": "base"},
                               {"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
    assert mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"] == "base"
    assert _census(app) is None


def test_streaming_census_assembles_the_tool_call_from_deltas(
        app, client, test_user, test_model, billable_key):
    _set_conditioning(app, test_model["id"], True)
    arguments = json.dumps({"todos": _FINISHED_TODOS})
    chunks = [
        _StreamChunk(delta=_Delta(tool_calls=[
            _ToolCall("todo_write", arguments[:20], index=0)])),
        _StreamChunk(delta=_Delta(tool_calls=[
            _ToolCall(None, arguments[20:], index=0)])),
        _StreamChunk(delta=_Delta(content="done"), finish_reason="tool_calls"),
        _StreamChunk(usage=_Usage()),
    ]
    # A delta fragment carries no name; the accumulator must keep the first one.
    chunks[1].choices[0].delta.tool_calls[0].function.name = None
    mock_cls, mock_client = _mock_openai(iter(chunks))
    with patch("lumen.blueprints.api.routes.openai.OpenAI", mock_cls):
        resp = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer lk_todo_tok"},
            json={"model": "test-model", "messages": _history(_OPEN_TODOS),
                  "tools": [_TODO_TOOL], "stream": True},
        )
        body = resp.get_data(as_text=True)  # drain the SSE generator so billing runs

    assert "data: [DONE]" in body
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "[lumen:todo-write]" in kwargs["messages"][0]["content"]

    census = _census(app)
    assert census["called"] is True
    assert census["all_completed"] is True
    assert census["finish_reason"] == "tool_calls"
