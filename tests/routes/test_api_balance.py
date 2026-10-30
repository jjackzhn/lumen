"""The balance endpoint reports the account's coins, not one key's usage."""

from datetime import datetime
from decimal import Decimal
from http import HTTPStatus

from lumen.extensions import db
from lumen.models.api_key import APIKey
from lumen.models.entity_balance import EntityBalance
from lumen.models.entity_stat import EntityStat
from lumen.services.crypto import hash_api_key


def _balance(client, token="balance-token"):
    return client.get("/v1/balance", headers={"Authorization": f"Bearer {token}"})


def _add(test_user, *rows):
    """Attach every row to the test user's entity and persist them."""
    for row in rows:
        row.entity_id = test_user["id"]
    db.session.add_all(rows)
    db.session.commit()


def test_balance_reports_account_coins_and_usage(app, client, test_user, restore_config):
    app.config["TOKEN_DEFAULTS"] = {"max": 100, "refresh": 10, "starting": 40}
    with app.app_context():
        _add(
            test_user,
            APIKey(name="balance", key_hash=hash_api_key("balance-token")),
            # The other key's usage is part of the account total, unlike /v1/usage.
            APIKey(name="other", key_hash=hash_api_key("other-token"), requests=5, cost=Decimal("4.000000")),
            EntityStat(
                requests=9, input_tokens=900, output_tokens=100,
                cost=Decimal("12.500000"), last_used_at=datetime(2026, 9, 28, 14, 0),
            ),
            EntityBalance(coins_left=Decimal("87.500000"), last_refill_at=datetime(2026, 9, 28, 13, 0)),
        )

    response = _balance(client)

    assert response.status_code == HTTPStatus.OK
    assert response.get_json() == {
        "coins_available": 87.5,
        "coins_spent": 12.5,
        "coins_limit": 100.0,
        "next_refill": "2026-09-28T14:00:00Z",
        "requests": 9,
        "tokens_used": 1000,
        "last_used_at": "2026-09-28T14:00:00Z",
    }
    # The pool is the account's, so every key of that account sees the same coins.
    assert _balance(client, "other-token").get_json()["coins_available"] == 87.5


def test_balance_without_a_pool_is_null_with_zero_usage(app, client, test_user):
    with app.app_context():
        _add(test_user, APIKey(name="bare", key_hash=hash_api_key("bare-token")))

    assert _balance(client, "bare-token").get_json() == {
        "coins_available": None,
        "coins_spent": 0.0,
        "coins_limit": None,
        "next_refill": None,
        "requests": 0,
        "tokens_used": 0,
        "last_used_at": None,
    }


def test_balance_reports_an_unlimited_pool_as_minus_two(app, client, test_user, restore_config):
    app.config["TOKEN_DEFAULTS"] = {"max": -2, "refresh": 0, "starting": 0}
    with app.app_context():
        _add(test_user, APIKey(name="unlimited", key_hash=hash_api_key("unlimited-token")))

    assert _balance(client, "unlimited-token").get_json() == {
        "coins_available": -2,
        "coins_spent": 0.0,
        "coins_limit": -2,
        "next_refill": None,
        "requests": 0,
        "tokens_used": 0,
        "last_used_at": None,
    }


def test_balance_requires_an_active_key(app, client, test_user):
    with app.app_context():
        _add(test_user, APIKey(
            name="revoked", key_hash=hash_api_key("revoked-token"),
            revoked_at=datetime(2026, 1, 1),
        ))

    assert client.get("/v1/balance").status_code == HTTPStatus.BAD_REQUEST
    assert _balance(client, "invalid").status_code == HTTPStatus.UNAUTHORIZED
    assert _balance(client, "revoked-token").status_code == HTTPStatus.UNAUTHORIZED
