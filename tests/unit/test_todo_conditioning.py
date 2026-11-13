"""The gateway-side todo_write reminder (services/todo_conditioning)."""
from lumen.services import todo_conditioning as tc


def _todo_tool(description="Record and update a task list."):
    return [{
        "type": "function",
        "function": {"name": "todo_write", "description": description,
                     "parameters": {"type": "object"}},
    }]


def test_reminder_appends_to_the_existing_system_message():
    messages = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "hi"}]
    messages, _ = tc.condition(messages, _todo_tool())
    assert len(messages) == 2
    # Appended at the end: a prefix that stays byte-stable keeps the upstream
    # prompt cache warm.
    assert messages[0]["content"].startswith("You are a coding agent.")
    assert tc.SYSTEM_REMINDER in messages[0]["content"]


def test_reminder_prepends_a_system_message_when_the_client_sent_none():
    messages = [{"role": "user", "content": "hi"}]
    messages, _ = tc.condition(messages, _todo_tool())
    assert messages[0] == {"role": "system", "content": tc.SYSTEM_REMINDER}
    assert messages[1]["role"] == "user"


def test_reminder_appends_to_a_content_part_list():
    messages = [{"role": "system", "content": [{"type": "text", "text": "base"}]}]
    messages, _ = tc.condition(messages, _todo_tool())
    parts = messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "base"}
    assert parts[-1] == {"type": "text", "text": tc.SYSTEM_REMINDER}


def test_tool_description_carries_the_reminder():
    tools = _todo_tool()
    _, tools = tc.condition([{"role": "user", "content": "hi"}], tools)
    assert tools[0]["function"]["description"].startswith("Record and update a task list.")
    assert tc.DESCRIPTION_REMINDER in tools[0]["function"]["description"]


def test_request_without_the_tool_is_untouched():
    messages = [{"role": "system", "content": "base"}]
    tools = [{"type": "function", "function": {"name": "bash", "description": "run"}}]
    out_messages, out_tools = tc.condition(messages, tools)
    assert out_messages[0]["content"] == "base"
    assert out_tools[0]["function"]["description"] == "run"


def test_request_without_tools_is_untouched():
    messages = [{"role": "system", "content": "base"}]
    out_messages, out_tools = tc.condition(messages, None)
    assert out_messages[0]["content"] == "base"
    assert out_tools is None


def test_conditioning_twice_adds_one_reminder():
    messages = [{"role": "system", "content": "base"}]
    tools = _todo_tool()
    tc.condition(messages, tools)
    tc.condition(messages, tools)
    assert messages[0]["content"].count(tc.MARKER) == 1
    assert tools[0]["function"]["description"].count(tc.MARKER) == 1


def test_a_reminder_the_client_already_sends_is_not_duplicated():
    messages = [{"role": "system", "content": f"base\n\n{tc.MARKER} already here"}]
    messages, _ = tc.condition(messages, _todo_tool())
    assert messages[0]["content"].count(tc.MARKER) == 1
