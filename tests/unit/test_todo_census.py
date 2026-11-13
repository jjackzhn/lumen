"""The per-request todo_write census (services/todo_census)."""
import json

from lumen.services import todo_census as tc


def _call(todos, name="todo_write"):
    return {"name": name, "arguments": json.dumps({"todos": todos})}


def _history(todos):
    return [
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


_TODOS = [
    {"content": "a", "status": "completed"},
    {"content": "b", "status": "in_progress"},
    {"content": "c", "status": "pending"},
]


def test_history_counts_read_the_list():
    assert tc.history_counts(_history(_TODOS)) == {"completed": 1, "in_progress": 1, "pending": 1}


def test_newest_list_wins():
    messages = _history([{"content": "a", "status": "pending"}])
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_2",
            "type": "function",
            "function": {"name": "todo_write",
                         "arguments": json.dumps({"todos": [{"content": "a", "status": "completed"}]})},
        }],
    })
    assert tc.history_counts(messages) == {"completed": 1, "in_progress": 0, "pending": 0}


def test_no_task_list_yields_none():
    assert tc.history_counts([{"role": "user", "content": "hi"}]) is None
    assert tc.history_counts(None) is None
    assert tc.census([{"role": "user", "content": "hi"}]) is None


def test_unknown_statuses_are_not_counted():
    counts = tc.history_counts(_history([
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "deleted"},
        {"content": "c"},
    ]))
    assert counts == {"completed": 1, "in_progress": 0, "pending": 0}


def test_malformed_arguments_are_ignored():
    messages = [{
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "todo_write", "arguments": "{not json"}}],
    }]
    assert tc.history_counts(messages) is None


def test_census_marks_a_list_left_open():
    row = tc.census(_history(_TODOS), [], "stop")
    assert row == {
        "items": 3, "completed": 1, "in_progress": 1, "pending": 1,
        "called": False, "all_completed": False, "finish_reason": "stop",
    }


def test_census_marks_a_finished_list():
    finished = [{"content": "a", "status": "completed"}]
    row = tc.census(_history(finished), [], "stop")
    assert row["all_completed"] is True and row["called"] is False


def test_the_responses_own_call_wins_over_history():
    row = tc.census(
        _history(_TODOS),
        [_call([{"content": "a", "status": "completed"}, {"content": "b", "status": "completed"},
                {"content": "c", "status": "completed"}])],
        "tool_calls",
    )
    assert row["called"] is True
    assert row["completed"] == 3 and row["all_completed"] is True
    assert row["finish_reason"] == "tool_calls"


def test_a_response_call_without_history_still_counts():
    row = tc.census([{"role": "user", "content": "hi"}],
                    [_call([{"content": "a", "status": "pending"}])], "tool_calls")
    assert row["called"] is True and row["items"] == 1 and row["all_completed"] is False


def test_an_unrelated_tool_call_is_not_a_todo_update():
    row = tc.census(_history(_TODOS), [_call(_TODOS, name="bash")], "tool_calls")
    assert row["called"] is False
    assert row["completed"] == 1
