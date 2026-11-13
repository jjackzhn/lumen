"""Measure whether a request's ``todo_write`` task list was kept up to date.

A model that opens a client-side task list and never updates it leaves the
user's progress panel showing finished work as still in progress. Lumen cannot
change what a model emits, but it sees both the list the client's history
carries and the tool calls one response made, which is enough to count the
behavior per model instead of inferring it from user reports.

Everything here is best effort: the census is observability, so a malformed
tool call yields no census rather than failing the request it describes.
"""
import json

TODO_TOOL_NAME = "todo_write"
_STATUSES = ("pending", "in_progress", "completed")


def _parse_arguments(raw):
    """Return the decoded tool-call arguments, or None when they are not an object."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _counts(todos):
    """Count each status in a todo list, or None when it is not a list.

    Entries without a known status are ignored rather than counted: a status
    this module does not know about is not evidence of progress.
    """
    if not isinstance(todos, list):
        return None
    counts = dict.fromkeys(_STATUSES, 0)
    for item in todos:
        status = item.get("status") if isinstance(item, dict) else None
        if status in counts:
            counts[status] += 1
    return counts


def history_counts(messages):
    """Return the counts of the newest ``todo_write`` list in the message history.

    The newest call wins: the tool replaces the whole list on every call, so the
    last one in history is the list the client is displaying. Returns None when
    the history carries no task list at all.
    """
    for message in reversed(messages or ()):
        if not isinstance(message, dict):
            continue
        for call in message.get("tool_calls") or ():
            function = call.get("function") if isinstance(call, dict) else None
            if not function or function.get("name") != TODO_TOOL_NAME:
                continue
            counts = _counts((_parse_arguments(function.get("arguments")) or {}).get("todos"))
            if counts is not None:
                return counts
    return None


def response_counts(tool_calls):
    """Return the counts of the last ``todo_write`` the response itself called.

    ``tool_calls`` are the response's calls as ``{"name", "arguments"}`` dicts:
    the caller flattens them because a streaming response assembles its calls
    from deltas. Returns None when the response called no task-list update.
    """
    counts = None
    for call in tool_calls or ():
        if call.get("name") != TODO_TOOL_NAME:
            continue
        parsed = _counts((_parse_arguments(call.get("arguments")) or {}).get("todos"))
        if parsed is not None:
            counts = parsed
    return counts


def census(messages, response_tool_calls=None, finish_reason=None):
    """Return the todo census for one request, or None when it carried no list.

    Keys:
      items          total entries in the newest list
      completed      entries marked completed in it
      in_progress    entries marked in_progress in it
      pending        entries marked pending in it
      called         whether this response called ``todo_write``
      all_completed  whether the list left behind is fully completed (the
                     response's own call when it made one, else the history's)
      finish_reason  the upstream finish reason, which tells a list left open
                     apart from one the model never got to update because
                     generation was cut short

    The response's call wins over the history's for the counts, because it is
    the state the client ends up displaying.
    """
    called = response_counts(response_tool_calls)
    effective = called if called is not None else history_counts(messages)
    if effective is None:
        return None
    total = sum(effective.values())
    return {
        "items": total,
        "completed": effective["completed"],
        "in_progress": effective["in_progress"],
        "pending": effective["pending"],
        "called": called is not None,
        "all_completed": total > 0 and effective["completed"] == total,
        "finish_reason": finish_reason,
    }
