"""Configuration and startup tests (offline)."""

from __future__ import annotations

from dataclasses import replace

import pytest

import config
from config import Settings, get_settings, load_dotenv
from src.service import AyurBotService
from tests.conftest import KB_PATH


@pytest.fixture()
def clean_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("AYURBOT_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_defaults(clean_env):
    s = get_settings(use_dotenv=False)
    assert s.port == 7860 and s.host == "127.0.0.1" and s.warnings == ()
    assert s.kb_path.name == "herbs.json" and s.kb_path.is_file()


def test_env_overrides(clean_env):
    clean_env.setenv("AYURBOT_PORT", "7901")
    clean_env.setenv("AYURBOT_TOP_K", "5")
    clean_env.setenv("AYURBOT_DISABLE_TRANSFORMER", "yes")
    clean_env.setenv("AYURBOT_MIN_LEXICAL_SCORE", "0.4")
    s = get_settings(use_dotenv=False)
    assert (s.port, s.top_k, s.disable_transformer, s.min_lexical_score) == (7901, 5, True, 0.4)


@pytest.mark.parametrize("key,value", [
    ("AYURBOT_PORT", "abc"), ("AYURBOT_PORT", "80"), ("AYURBOT_TOP_K", "0"),
    ("AYURBOT_MIN_SEMANTIC_SCORE", "2"), ("AYURBOT_OFFLINE", "maybe"),
])
def test_invalid_values_warn_and_use_defaults(clean_env, key, value):
    clean_env.setenv(key, value)
    s = get_settings(use_dotenv=False)
    assert s.warnings and key in s.warnings[0]
    assert getattr(s, key.removeprefix("AYURBOT_").lower()) == getattr(Settings(), key.removeprefix("AYURBOT_").lower())


def test_relative_kb_path_is_project_relative(clean_env):
    clean_env.setenv("AYURBOT_KB_PATH", "data/herbs.json")
    assert get_settings(use_dotenv=False).kb_path == KB_PATH.resolve()


def test_dotenv_does_not_override_real_env(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text("# comment\nAYURBOT_PORT=7999\nAYURBOT_TOP_K='4'\n", encoding="utf-8")
    clean_env.setenv("AYURBOT_PORT", "7800")
    applied = load_dotenv(env)
    assert applied == {"AYURBOT_TOP_K": "4"}
    s = get_settings(use_dotenv=False)
    assert s.port == 7800 and s.top_k == 4


def test_service_lexical_mode_when_disabled(fake_factory):
    s = replace(Settings(), disable_transformer=True)
    svc = AyurBotService(s, model_factory=fake_factory)
    assert svc.status.kb_ok and svc.status.retrieval_mode == "lexical"
    assert svc.status.model_state == "disabled" and fake_factory.calls["n"] == 0
    assert svc.ask("What is Tulsi?").status == "ok"


def test_service_semantic_mode_with_model(fake_factory):
    svc = AyurBotService(Settings(), model_factory=fake_factory)
    assert svc.status.retrieval_mode == "semantic" and svc.status.model_state == "ready"
    svc.ask("Tulsi"); svc.ask("Neem")
    assert fake_factory.calls["n"] == 1  # model loaded once for the whole app


def test_service_survives_model_failure(failing_factory):
    svc = AyurBotService(Settings(), model_factory=failing_factory)
    assert svc.status.retrieval_mode == "lexical" and svc.status.model_state == "failed"
    for _ in range(3):
        assert svc.ask("Does Neem pacify Pitta?").status == "ok"
    assert failing_factory.calls["n"] == 1


def test_service_with_broken_kb_does_not_crash(tmp_path, fake_factory):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    svc = AyurBotService(replace(Settings(), kb_path=bad), model_factory=fake_factory)
    assert not svc.status.kb_ok and "Invalid JSON" in svc.status.kb_message
    r = svc.ask("Tulsi")
    assert r.status == "error" and "failed to load" in r.answer
    assert fake_factory.calls["n"] == 0


def test_service_catches_unexpected_errors(fake_factory, monkeypatch):
    svc = AyurBotService(replace(Settings(), disable_transformer=True), model_factory=fake_factory)
    monkeypatch.setattr(svc.engine, "respond", lambda q: 1 / 0)
    r = svc.ask("Tulsi")
    assert r.status == "error" and "ZeroDivisionError" in r.answer


def test_concurrent_queries_are_consistent(fake_factory):
    from concurrent.futures import ThreadPoolExecutor

    svc = AyurBotService(Settings(), model_factory=fake_factory)
    queries = ["Tulsi and Kapha", "Neem for Kushtha", "What is Agni?", "Brahmi 2 g"] * 10
    expected = {q: svc.ask(q).answer for q in set(queries)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda q: (q, svc.ask(q).answer), queries))
    assert all(a == expected[q] for q, a in answers)


def test_project_root_constant():
    assert (config.PROJECT_ROOT / "app.py").is_file()
