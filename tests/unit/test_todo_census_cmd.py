"""``flask todo-census``: the per-model task-list completion table.

The counts come from ``request_logs.todo_census``, written per request by
billing, so the command only has to group them and apply the "stuck" rule: a
list left open by a response that ended its turn normally.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from lumen.commands import todo_census_cmd
from lumen.extensions import db
from lumen.models.model_config import ModelConfig
from lumen.models.request_log import RequestLog


@pytest.fixture
def model_id(app):
    with app.app_context():
        model = ModelConfig(model_name="census-model", input_cost_per_million=1.0,
                            output_cost_per_million=1.0)
        db.session.add(model)
        db.session.commit()
        return model.id


def _log(app, model_id, census, age=timedelta(0)):
    """A request_logs row carrying a census, aged back from now."""
    with app.app_context():
        db.session.add(RequestLog(
            time=datetime.now(timezone.utc) - age,
            model_config_id=model_id, source="api",
            input_tokens=10, output_tokens=5, cost=Decimal("0.5"), duration=0.0,
            outcome="ok", todo_census=census,
        ))
        db.session.commit()


def _census(completed, finish_reason="stop", called=False, items=2):
    return {
        "items": items,
        "completed": completed,
        "in_progress": items - completed,
        "pending": 0,
        "called": called,
        "all_completed": completed == items,
        "finish_reason": finish_reason,
    }


def _run(app, *args):
    return app.test_cli_runner().invoke(todo_census_cmd, list(args))


def test_the_command_is_registered_on_the_app(app):
    # The other tests invoke the command object directly, which would keep
    # passing if create_app() forgot to register it on the CLI group.
    result = app.test_cli_runner().invoke(args=["todo-census", "--days", "1"])
    assert result.exit_code == 0, result.output
    assert "No such command" not in result.output


def test_reports_each_model_with_its_stuck_share(app, model_id):
    _log(app, model_id, _census(2))
    _log(app, model_id, _census(1))
    _log(app, model_id, _census(1))

    result = _run(app, "--days", "7")
    assert result.exit_code == 0
    assert "census-model" in result.output
    # 2 requests left a list open and both ended normally, so 2 of 3 are stuck.
    assert "(67%)" in result.output


def test_a_truncated_response_is_open_but_not_stuck(app, model_id):
    _log(app, model_id, _census(1, finish_reason="length"))

    result = _run(app, "--days", "7")
    assert result.exit_code == 0
    assert "(0%)" in result.output


def test_a_response_that_ended_on_a_tool_call_is_not_stuck(app, model_id):
    _log(app, model_id, _census(1, finish_reason="tool_calls"))

    result = _run(app, "--days", "7")
    assert "(0%)" in result.output


def test_rows_outside_the_window_are_ignored(app, model_id):
    _log(app, model_id, _census(1), age=timedelta(days=10))

    result = _run(app, "--days", "7")
    assert result.exit_code == 0
    assert "No task-list requests" in result.output


def test_requests_without_a_census_are_ignored(app, model_id):
    with app.app_context():
        db.session.add(RequestLog(
            time=datetime.now(timezone.utc), model_config_id=model_id, source="api",
            input_tokens=10, output_tokens=5, cost=Decimal("0.5"), duration=0.0,
            outcome="ok", todo_census=None,
        ))
        db.session.commit()

    result = _run(app)
    assert "No task-list requests" in result.output
