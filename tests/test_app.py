"""UI wiring tests. Skipped automatically if Gradio is not installed."""

from __future__ import annotations

from dataclasses import replace

import pytest

gr = pytest.importorskip("gradio")

import app  # noqa: E402
from config import Settings  # noqa: E402
from src.response_engine import ENTITY_COLUMNS  # noqa: E402
from src.service import AyurBotService  # noqa: E402


@pytest.fixture()
def service(fake_factory):
    return AyurBotService(replace(Settings(), max_history_messages=6), model_factory=fake_factory)


def test_ui_builds(service):
    demo = app.build_ui(service)
    assert isinstance(demo, gr.Blocks)


def test_two_stage_chat_flow(service):
    queue_user, answer, clear_all = app.make_handlers(service)
    box, history, pending = queue_user("Does Tulsi pacify Kapha?", [])
    assert box == "" and pending == "Does Tulsi pacify Kapha?" and history[-1]["role"] == "user"
    history, highlighted, entities, evidence, status = answer(pending, history)
    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "pacifying **Kapha**" in history[-1]["content"]
    assert ("Tulsi", "HERB") in highlighted
    assert list(entities.columns) == ENTITY_COLUMNS and len(entities) == 2
    assert "Semantic search" in status
    cleared = clear_all()
    assert cleared[0] == [] and cleared[2].empty


def test_empty_submit_keeps_history(service):
    queue_user, answer, _ = app.make_handlers(service)
    old = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    _, history, pending = queue_user("   ", old)
    assert history == old and pending == ""


def test_history_is_bounded_and_normalised(service):
    queue_user, answer, _ = app.make_handlers(service)
    history = []
    for i in range(10):
        _, history, pending = queue_user(f"What is Neem? {i}", history)
        history = answer(pending, history)[0]
    assert len(history) == 6


def test_gradio6_content_blocks_are_normalised():
    history = [
        {"role": "user", "metadata": None, "content": [{"text": "What is Vata?", "type": "text"}], "options": None},
        {"role": "assistant", "content": [{"type": "text", "text": "Answer"}]},
        ("legacy q", "legacy a"),
    ]
    assert app.normalize_history(history, 10) == [
        {"role": "user", "content": "What is Vata?"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "legacy q"},
        {"role": "assistant", "content": "legacy a"},
    ]


def test_status_shows_fallback(failing_factory):
    svc = AyurBotService(Settings(), model_factory=failing_factory)
    text = app.status_markdown(svc)
    assert "Lexical search" in text and "simulated" in text


def test_status_shows_kb_error(tmp_path, fake_factory):
    bad = tmp_path / "x.json"
    bad.write_text("[]", encoding="utf-8")
    svc = AyurBotService(replace(Settings(), kb_path=bad), model_factory=fake_factory)
    assert "Knowledge base error" in app.status_markdown(svc)


def test_find_free_port_skips_busy_port():
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        port = busy.getsockname()[1]
        assert app.find_free_port("127.0.0.1", port, 5) not in (None, port)


def test_legend_lists_all_labels():
    html = app.legend_html()
    for label in app.ENTITY_COLORS:
        assert label in html
