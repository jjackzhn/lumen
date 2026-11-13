"""Record the todo_write census and gate the completion reminder

Revision ID: a7b8c9d0e1f2
Revises: e3f4a5b6c7d8
Create Date: 2026-11-13 00:00:00.000000

Two halves of one change. ``request_logs.todo_census`` records, for a request
that carried a client-side ``todo_write`` task list, the counts of the list the
response left behind — the measurement that makes "models leave task lists
stuck" visible per model instead of a user report. ``model_configs.todo_
conditioning`` is the per-model gate for the gateway-side reminder that asks the
model to finish with every item completed; it defaults to false so the effect is
measured before any model is switched over.

No backfill. The census describes a request's own response, which historical
rows never captured, and the gate is a deployment decision rather than a value
any earlier row implies.
"""

import sqlalchemy as sa
from alembic import op

revision = "a7b8c9d0e1f2"
down_revision = "e3f4a5b6c7d8"
branch_labels = None
depends_on = None


def _is_postgresql():
    return op.get_bind().dialect.name == "postgresql"


def upgrade():
    # Default first, then NOT NULL in the same statement: a populated
    # hypertable rejects a bare ADD COLUMN ... NOT NULL (see y9z0a1b2c3d4).
    with op.batch_alter_table("model_configs") as batch_op:
        batch_op.add_column(
            sa.Column(
                "todo_conditioning",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )

    # Nullable: NULL means the request carried no task list, or ended before its
    # response was observed. That is a different claim from a list left open.
    with op.batch_alter_table("request_logs") as batch_op:
        batch_op.add_column(sa.Column("todo_census", sa.JSON(), nullable=True))

    if _is_postgresql():
        op.execute(
            "COMMENT ON COLUMN model_configs.todo_conditioning IS "
            "$comment$Append the todo_write completion reminder to the system "
            "prompt and tool description; off by default$comment$"
        )
        op.execute(
            "COMMENT ON COLUMN request_logs.todo_census IS "
            "$comment$Task-list snapshot for this request: counts from the "
            "latest todo_write in history plus whether this response called "
            "it; NULL when the request carried no task list or its response "
            "was not observed$comment$"
        )


def downgrade():
    with op.batch_alter_table("request_logs") as batch_op:
        batch_op.drop_column("todo_census")
    with op.batch_alter_table("model_configs") as batch_op:
        batch_op.drop_column("todo_conditioning")
