"""Gateway-side reminder that keeps a model from leaving a task list open.

Some models open a client-side ``todo_write`` task list and never send the
final update, so the client's progress panel shows finished work as still in
progress. Instructions the client ships in its own system prompt do not
reliably reach the model, so this appends one constant reminder to the system
prompt and to the tool's own description, where a model weighs it more heavily.

Both reminders are appended at the end of their block and are byte-identical on
every request, so after the first one the upstream prefix cache still hits.
Nothing is added to a request that carries no ``todo_write`` tool.
"""

TODO_TOOL_NAME = "todo_write"

#: Presence marker. A reminder already carrying it is never appended again, so
#: a client that sends its own copy gets one reminder, not two.
MARKER = "[lumen:todo-write]"

SYSTEM_REMINDER = (
    f"{MARKER} Keep the `{TODO_TOOL_NAME}` task list current: whenever a task's "
    "status changes, call the tool again with the COMPLETE list and the new "
    "statuses, and finish the work with every item marked `completed`."
)

DESCRIPTION_REMINDER = (
    f"{MARKER} Call this tool one final time when the work is finished, with "
    "every item marked `completed`."
)


def _has_marker(value):
    return isinstance(value, str) and MARKER in value


def _carries_todo_tool(tools):
    for tool in tools or ():
        function = tool.get("function") if isinstance(tool, dict) else None
        if function and function.get("name") == TODO_TOOL_NAME:
            return True
    return False


def _condition_system_message(messages):
    """Append the reminder to the leading system message, or prepend one."""
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content")
        if isinstance(content, str):
            if not _has_marker(content):
                messages[0] = {**messages[0], "content": f"{content}\n\n{SYSTEM_REMINDER}"}
        elif isinstance(content, list):
            # A content-part list: the reminder rides as a final text part so
            # multimodal blocks keep their order.
            if not any(_has_marker(part.get("text")) for part in content if isinstance(part, dict)):
                messages[0] = {**messages[0], "content": [*content, {"type": "text", "text": SYSTEM_REMINDER}]}
    else:
        messages.insert(0, {"role": "system", "content": SYSTEM_REMINDER})


def _condition_tools(tools):
    """Append the reminder to the ``todo_write`` description."""
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not function or function.get("name") != TODO_TOOL_NAME:
            continue
        description = function.get("description") or ""
        if not _has_marker(description):
            function["description"] = f"{description}\n\n{DESCRIPTION_REMINDER}".strip()


def condition(messages, tools):
    """Apply the reminder to ``messages`` and ``tools``, which are updated in place.

    The call sites build both from the request's own JSON, so updating them in
    place is what makes the reminder reach the upstream call. Returns the pair
    so a call site can write ``messages, tools = condition(messages, tools)``.

    A request with no ``todo_write`` tool, and one already carrying the marker,
    is left exactly as it arrived.
    """
    if not _carries_todo_tool(tools):
        return messages, tools
    _condition_system_message(messages)
    _condition_tools(tools)
    return messages, tools
