"""Tests for the SQLite-backed session_store.py. The isolate_db
fixture (conftest.py) points every test at its own temp file, so these
never touch real dev data and never leak into each other.
"""
import pytest

from orchestrator.extraction import session_store
from orchestrator.extraction.models import Plan

SAMPLE_PLAN = Plan.model_validate({
    "name": "test-plan",
    "summary": "A test plan.",
    "steps": [{"id": "StepA", "service": "S3", "description": "does something"}],
})


def test_create_session_returns_unique_ids():
    s1 = session_store.create_session()
    s2 = session_store.create_session()
    assert s1.session_id != s2.session_id


def test_get_unknown_session_returns_none():
    assert session_store.get_session("does-not-exist") is None


def test_append_message_to_unknown_session_raises():
    with pytest.raises(KeyError):
        session_store.append_message("does-not-exist", "user", "hi")


def test_history_persists_and_orders_correctly():
    session = session_store.create_session()
    session_store.append_message(session.session_id, "user", "first")
    session_store.append_message(session.session_id, "assistant", "second")
    session_store.append_message(session.session_id, "user", "third")

    reloaded = session_store.get_session(session.session_id)
    contents = [m["content"] for m in reloaded.history]
    assert contents == ["first", "second", "third"]


def test_history_survives_simulated_restart():
    """The real point of moving to SQLite: a brand new call to
    get_session (a fresh connection under the hood, same as after an
    API process restart) still sees data written earlier.
    """
    session = session_store.create_session()
    session_store.append_message(session.session_id, "user", "remember me")

    # No shared connection or in-memory object reused here --
    # get_session opens its own fresh connection each call.
    reloaded = session_store.get_session(session.session_id)
    assert reloaded.history == [{"role": "user", "content": "remember me"}]


def test_latest_plan_starts_none():
    session = session_store.create_session()
    assert session.latest_plan is None


def test_set_latest_plan_persists_and_round_trips():
    session = session_store.create_session()
    session_store.set_latest_plan(session.session_id, SAMPLE_PLAN)

    reloaded = session_store.get_session(session.session_id)
    assert reloaded.latest_plan is not None
    assert reloaded.latest_plan.name == "test-plan"
    assert reloaded.latest_plan.steps[0].id == "StepA"


def test_set_latest_plan_none_clears_it():
    session = session_store.create_session()
    session_store.set_latest_plan(session.session_id, SAMPLE_PLAN)
    session_store.set_latest_plan(session.session_id, None)

    reloaded = session_store.get_session(session.session_id)
    assert reloaded.latest_plan is None


def test_set_latest_plan_unknown_session_raises():
    with pytest.raises(KeyError):
        session_store.set_latest_plan("does-not-exist", SAMPLE_PLAN)


def test_two_sessions_do_not_share_history():
    s1 = session_store.create_session()
    s2 = session_store.create_session()
    session_store.append_message(s1.session_id, "user", "session one message")
    session_store.append_message(s2.session_id, "user", "session two message")

    r1 = session_store.get_session(s1.session_id)
    r2 = session_store.get_session(s2.session_id)
    assert r1.history == [{"role": "user", "content": "session one message"}]
    assert r2.history == [{"role": "user", "content": "session two message"}]