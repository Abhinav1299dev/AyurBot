"""AyurBot web application (Gradio).

Run from the project folder:

    .\\.venv\\Scripts\\python.exe app.py      (Windows)
    python app.py                            (any platform, inside the venv)

Then open http://127.0.0.1:7860 in a browser.
"""

from __future__ import annotations

import html
import logging
import os
import socket
import sys
from typing import Any

# Keep startup quiet and offline-friendly before Gradio is imported.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from config import Settings, get_settings  # noqa: E402
from src.knowledge_base import ENTITY_LABELS  # noqa: E402
from src.response_engine import DISCLAIMER, ENTITY_COLUMNS, EVIDENCE_COLUMNS  # noqa: E402
from src.service import AyurBotService  # noqa: E402

log = logging.getLogger("ayurbot")

ENTITY_COLORS = {
    "HERB": "#9fd8a9",
    "BOTANICAL_NAME": "#b9c6f5",
    "DOSHA": "#f3d27a",
    "CONDITION_TERM": "#f4b2ae",
    "TRADITIONAL_PROPERTY": "#d8c0f0",
    "DOSAGE_MENTION": "#a9d8f0",
    "AYURVEDIC_CONCEPT": "#f5c79a",
}
ENTITY_DESCRIPTIONS = {
    "HERB": "Herb name, synonym or English name",
    "BOTANICAL_NAME": "Latin botanical name",
    "DOSHA": "Vata, Pitta or Kapha",
    "CONDITION_TERM": "Classical condition term",
    "TRADITIONAL_PROPERTY": "Rasa, virya or traditional action",
    "DOSAGE_MENTION": "Quantity you mentioned (not advice)",
    "AYURVEDIC_CONCEPT": "Concept such as Agni or Prakriti",
}
CSS = """
:root {
  --ab-sage-900: #2f4236; --ab-sage-800: #3b5444; --ab-sage-700: #4b6b55; --ab-sage-500: #7a9a7e;
  --ab-sage-200: #d6e2d3; --ab-sage-100: #e9f0e6; --ab-cream: #f6f1e6; --ab-cream-card: #fffcf6;
  --ab-cream-deep: #efe7d6; --ab-line: #e5ddcb; --ab-ink: #263228; --ab-muted: #6a756b;
  --ab-terracotta: #b8743a; --ab-terracotta-soft: #f7ead9;
  --ab-vata: #7b93ab; --ab-pitta: #c46a34; --ab-kapha: #6b9467;
  --ab-radius: 16px; --ab-shadow: 0 1px 2px rgba(47,66,54,.05), 0 8px 24px rgba(47,66,54,.06);
}
body, .gradio-container { background: var(--ab-cream) !important; color: var(--ab-ink); }
.gradio-container { max-width: 1440px !important; margin: 0 auto !important; padding: 16px 20px 24px !important; }
footer { display: none !important; }
@keyframes abFade { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ---------- Sidebar ---------- */
#ab-sidebar { background: linear-gradient(180deg, var(--ab-sage-900), var(--ab-sage-800)) !important;
  border-radius: 20px !important; padding: 20px 14px !important; position: sticky; top: 16px;
  align-self: flex-start; min-height: calc(100vh - 48px); gap: 6px !important; border: none !important; }
#ab-brand { display: flex; align-items: center; gap: 12px; padding: 0 6px 16px;
  border-bottom: 1px solid rgba(246,241,230,.14); margin-bottom: 10px; }
#ab-brand h1 { font-family: Georgia, "Palatino Linotype", serif; font-size: 1.45rem; font-weight: 600;
  color: #fbf6ea; margin: 0; line-height: 1.1; }
#ab-brand p { margin: 3px 0 0; font-size: .78rem; color: #c5d3c4; }
.ab-nav-label p { color: #a9bba9 !important; font-size: .7rem !important; letter-spacing: .08em;
  text-transform: uppercase; margin: 8px 8px 2px !important; font-weight: 600; }
#ab-sidebar button.ab-nav { justify-content: flex-start !important; text-align: left !important;
  background: transparent !important; color: #e6ede4 !important; border: 1px solid transparent !important;
  box-shadow: none !important; font-weight: 500 !important; font-size: .93rem !important;
  border-radius: 12px !important; padding: 10px 14px !important; transition: background .18s, color .18s; }
#ab-sidebar button.ab-nav:hover { background: rgba(246,241,230,.10) !important; }
#ab-sidebar button.ab-nav.primary { background: var(--ab-cream) !important; color: var(--ab-sage-900) !important;
  font-weight: 650 !important; }
#ab-sidebar button.ab-nav:focus-visible { outline: 2px solid #e8c48a !important; outline-offset: 2px; }
#ab-side-foot { margin-top: auto; padding: 14px 8px 0; border-top: 1px solid rgba(246,241,230,.14);
  font-size: .78rem; color: #c5d3c4; line-height: 1.5; }
.ab-pill { display: inline-flex; align-items: center; gap: 7px; font-size: .76rem; font-weight: 600;
  padding: 5px 10px; border-radius: 999px; background: rgba(0,0,0,.18); color: #f3f6f1; margin: 0 4px 6px 0; }
.ab-dot { width: 8px; height: 8px; border-radius: 50%; background: #93d4a1; }
.ab-dot.warn { background: #e9bf5c; } .ab-dot.err { background: #ec7c70; }

/* ---------- Screens and page headers ---------- */
.ab-screen { animation: abFade .28s ease both; gap: 14px !important; }
.ab-page-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  padding: 4px 4px 2px; }
.ab-page-head h2 { font-family: Georgia, "Palatino Linotype", serif; font-size: 1.75rem; font-weight: 600;
  color: var(--ab-sage-900); margin: 0; letter-spacing: -.01em; }
.ab-page-head p { margin: 4px 0 0; color: var(--ab-muted); font-size: .93rem; max-width: 70ch; }
.ab-eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  color: var(--ab-terracotta); }

/* ---------- Cards ---------- */
.ab-card { background: var(--ab-cream-card) !important; border: 1px solid var(--ab-line) !important;
  border-radius: var(--ab-radius) !important; padding: 18px !important; box-shadow: var(--ab-shadow); }
.ab-section-title h3 { margin: 0 0 3px !important; font-family: Georgia, serif; font-size: 1.1rem !important;
  font-weight: 600; color: var(--ab-sage-900); }
.ab-section-title p { margin: 0 !important; color: var(--ab-muted); font-size: .86rem; }
#ab-disclaimer { display: flex; gap: 12px; align-items: flex-start; background: var(--ab-terracotta-soft);
  border-left: 4px solid var(--ab-terracotta); border-radius: 12px; padding: 12px 16px; color: #4d3a26;
  font-size: .86rem; line-height: 1.55; }
#ab-disclaimer svg { flex: none; margin-top: 1px; }

/* ---------- Welcome ---------- */
.ab-hero { position: relative; overflow: hidden; border-radius: 22px; padding: 34px 34px 30px;
  background: linear-gradient(135deg, var(--ab-sage-100) 0%, var(--ab-cream-card) 60%, var(--ab-cream-deep) 100%);
  border: 1px solid var(--ab-line); }
.ab-hero h2 { font-family: Georgia, "Palatino Linotype", serif; font-size: 2.3rem; line-height: 1.15;
  color: var(--ab-sage-900); margin: 8px 0 10px; max-width: 20ch; font-weight: 600; }
.ab-hero p { color: var(--ab-muted); font-size: 1rem; line-height: 1.6; max-width: 62ch; margin: 0; }
.ab-hero svg.ab-motif { position: absolute; right: -30px; bottom: -50px; opacity: .22; }
#ab-cta button { border-radius: 999px !important; }
.ab-feature-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; }
.ab-feature { background: var(--ab-cream-card); border: 1px solid var(--ab-line); border-radius: 14px; padding: 16px;
  transition: transform .18s ease, box-shadow .18s ease; }
.ab-feature:hover { transform: translateY(-2px); box-shadow: var(--ab-shadow); }
.ab-feature .ic { width: 36px; height: 36px; border-radius: 10px; display: grid; place-items: center;
  background: var(--ab-sage-100); font-size: 1.1rem; margin-bottom: 10px; }
.ab-feature h4 { margin: 0 0 4px; font-size: .98rem; color: var(--ab-sage-900); }
.ab-feature p { margin: 0; font-size: .85rem; color: var(--ab-muted); line-height: 1.5; }
.ab-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; counter-reset: s; }
.ab-step { padding: 12px 14px 12px 46px; position: relative; font-size: .86rem; color: var(--ab-ink); line-height: 1.45; }
.ab-step::before { counter-increment: s; content: counter(s); position: absolute; left: 10px; top: 12px; width: 26px;
  height: 26px; border-radius: 50%; background: var(--ab-sage-700); color: #fff; display: grid; place-items: center;
  font-size: .8rem; font-weight: 700; }
.ab-step b { display: block; color: var(--ab-sage-900); margin-bottom: 2px; }

/* ---------- Stats and metrics ---------- */
.ab-stats, .ab-metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.ab-stat, .ab-metric { background: var(--ab-cream-card); border: 1px solid var(--ab-line); border-radius: 14px;
  padding: 14px 16px; }
.ab-stat b, .ab-metric b { display: block; font-family: Georgia, serif; font-size: 1.65rem; color: var(--ab-sage-800);
  font-weight: 600; line-height: 1.1; }
.ab-stat span, .ab-metric span { font-size: .8rem; color: var(--ab-muted); }
.ab-metric small { display: block; font-size: .74rem; color: var(--ab-muted); margin-top: 4px; }
.ab-bar { height: 6px; border-radius: 99px; background: var(--ab-sage-100); margin-top: 8px; overflow: hidden; }
.ab-bar i { display: block; height: 100%; background: var(--ab-sage-700); border-radius: 99px; }

/* ---------- Chat ---------- */
#ab-chat { border: 1px solid var(--ab-line) !important; border-radius: 14px !important; background: #fffefb !important; }
#ab-input textarea { font-size: .98rem !important; background: #fffefb !important; }
#ab-send { min-height: 46px; align-self: stretch; }
.ab-topic-label p { margin: 10px 0 4px !important; font-size: .76rem; font-weight: 650; color: var(--ab-muted); }
button.ab-chip { border-radius: 999px !important; background: var(--ab-sage-100) !important;
  border: 1px solid transparent !important; color: var(--ab-sage-900) !important; font-weight: 500 !important;
  font-size: .83rem !important; box-shadow: none !important; transition: border-color .15s, background .15s; }
button.ab-chip:hover { border-color: var(--ab-sage-500) !important; background: var(--ab-sage-200) !important; }
#ab-legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px 14px;
  font-size: .79rem; color: var(--ab-muted); padding: 10px 12px; background: var(--ab-sage-100);
  border-radius: 12px; margin: 6px 0 4px; }
#ab-legend b { color: var(--ab-ink); font-weight: 600; margin-right: 4px; font-size: .73rem; }
#ab-legend span.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px;
  border: 1px solid rgba(0,0,0,.15); }
#ab-status { font-size: .86rem; line-height: 1.55; }

/* ---------- Dosha cards ---------- */
.ab-dosha-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }
.ab-dosha-card { background: var(--ab-cream-card); border: 1px solid var(--ab-line); border-top: 5px solid var(--c);
  border-radius: 16px; padding: 18px; transition: transform .18s ease, box-shadow .18s ease; }
.ab-dosha-card:hover { transform: translateY(-2px); box-shadow: var(--ab-shadow); }
.ab-dosha-card h4 { margin: 0; font-family: Georgia, serif; font-size: 1.3rem; color: var(--ab-sage-900); }
.ab-dosha-card .el { display: inline-block; margin: 6px 0 10px; font-size: .76rem; font-weight: 600; padding: 4px 10px;
  border-radius: 999px; background: color-mix(in srgb, var(--c) 16%, transparent); color: var(--ab-ink); }
.ab-dosha-card p { font-size: .88rem; color: var(--ab-ink); line-height: 1.55; margin: 0 0 8px; }
.ab-dosha-card ul { margin: 0 0 10px; padding-left: 18px; font-size: .85rem; color: var(--ab-muted); }
.ab-dosha-card .herbs { font-size: .84rem; background: var(--ab-cream); border-radius: 10px; padding: 10px 12px;
  line-height: 1.6; }

/* ---------- Responsive ---------- */
@media (max-width: 900px) {
  .gradio-container { padding: 10px !important; }
  #ab-sidebar { position: static; min-height: 0; padding: 14px !important; }
  #ab-nav { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px; }
  #ab-side-foot { display: none; }
  .ab-hero { padding: 24px 20px; } .ab-hero h2 { font-size: 1.7rem; }
  .ab-page-head h2 { font-size: 1.4rem; }
  .ab-card { padding: 12px !important; }
}
"""

# The design is a light sage-and-cream palette, so pages always open in light mode.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

SCREENS = [
    ("welcome", "🏡", "Welcome"),
    ("chat", "💬", "Ask AyurBot"),
    ("herbs", "🌿", "Herb Explorer"),
    ("doshas", "🔥", "Dosha Guide"),
    ("kb", "📚", "Knowledge Base & System"),
    ("eval", "📊", "Performance Evaluation"),
]

TOPICS = {
    "Herbs": ["What is Tulsi?", "Tell me about Withania somnifera", "Compare Neem and Guduchi for skin disease"],
    "Doshas": ["What is Vata?", "Does Haridra pacify Kapha?", "Which herbs may aggravate Vata?"],
    "Classical terms": ["Which herbs are traditionally described for cough?", "What does Medhya mean?", "What is Jwara?"],
    "Concepts": ["What is the difference between Prakriti and Vikriti?", "What is Agni?", "तुलसी क्या है?"],
}



# ---------------------------------------------------------------- helpers
def message_text(content: Any) -> str:
    """Extract plain text from any Gradio chat message content shape."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        return message_text(content.get("content"))
    if isinstance(content, (list, tuple)):
        return "\n".join(t for t in (message_text(c) for c in content) if t)
    return str(content)


def normalize_history(history: Any, max_messages: int) -> list[dict[str, str]]:
    """Return a clean, bounded ``[{role, content}]`` list."""
    clean: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict) and item.get("role") in ("user", "assistant"):
            clean.append({"role": item["role"], "content": message_text(item.get("content"))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:  # legacy tuple format
            if item[0] is not None:
                clean.append({"role": "user", "content": message_text(item[0])})
            if item[1] is not None:
                clean.append({"role": "assistant", "content": message_text(item[1])})
    return clean[-max_messages:]


def legend_html() -> str:
    items = "".join(
        f'<div><span class="sw" style="background:{ENTITY_COLORS[label]}"></span>'
        f"<b>{label}</b> {html.escape(ENTITY_DESCRIPTIONS[label])}</div>"
        for label in ENTITY_LABELS
    )
    return f'<div id="ab-legend" role="list" aria-label="Entity colour legend">{items}</div>'


def status_markdown(service: AyurBotService, last=None) -> str:
    st = service.refresh_status()
    if not st.kb_ok:
        return f"**Knowledge base error.** The chatbot cannot answer until this is fixed:\n\n```\n{st.kb_message}\n```"
    mode = service.mode_label
    model_line = {
        "ready": f"Ready. {st.model_detail}",
        "disabled": f"Off. {st.model_detail} Lexical search is answering.",
        "failed": f"Unavailable, so lexical fallback is answering. Reason: {st.model_detail}",
    }.get(st.model_state, st.model_state)
    counts = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(st.kb_counts.items()))
    lines = [
        f"**Retrieval:** {mode}",
        f"**Transformer model:** {model_line}",
        f"**Knowledge base:** {st.kb_message} ({counts})",
        f"**NER:** rule-based, {st.ner_terms} dictionary terms plus dosage patterns",
        f"**Startup time:** {st.startup_seconds:.1f} s",
    ]
    if last is not None:
        lines.append(f"**Last answer:** {last.elapsed_ms:.0f} ms using {last.retrieval_mode or 'n/a'} retrieval")
        if last.retrieval_note:
            lines.append(f"**Note:** {last.retrieval_note}")
    warnings = st.config_warnings + st.kb_warnings
    if warnings:
        lines.append("**Warnings:** " + " ".join(warnings))
    return "\n\n".join(lines)


LEAF_SVG = (
    '<svg width="40" height="40" viewBox="0 0 48 48" aria-hidden="true">'
    '<rect width="48" height="48" rx="13" fill="#f6f1e6"/>'
    '<path d="M14 34c0-12 8-20 22-21-1 14-9 22-21 22" fill="#7a9a7e"/>'
    '<path d="M14 35c5-7 10-12 17-17" stroke="#2f4236" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
    '</svg>'
)
SHIELD_SVG = (
    '<svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 '
    '4.6-1.6 8-6 8-11V5l-8-3z" fill="#b8743a"/><path d="M11 7h2v6h-2zM11 15h2v2h-2z" fill="#fff"/></svg>'
)
MOTIF_SVG = (
    '<svg class="ab-motif" width="300" height="300" viewBox="0 0 200 200" aria-hidden="true">'
    + "".join(
        f'<ellipse cx="100" cy="55" rx="16" ry="45" fill="none" stroke="#4b6b55" stroke-width="1.3" '
        f'transform="rotate({angle} 100 100)"/>'
        for angle in range(0, 360, 30)
    )
    + '<circle cx="100" cy="100" r="22" fill="none" stroke="#4b6b55" stroke-width="1.3"/></svg>'
)


def mode_pill_html(service: AyurBotService) -> str:
    st = service.refresh_status()
    if not st.kb_ok:
        return '<span class="ab-pill"><span class="ab-dot err"></span>Knowledge base error</span>'
    if st.retrieval_mode == "semantic":
        return '<span class="ab-pill"><span class="ab-dot"></span>Semantic search</span>'
    return '<span class="ab-pill"><span class="ab-dot warn"></span>Lexical fallback</span>'


def page_head(eyebrow: str, title: str, subtitle: str) -> str:
    return (f'<div class="ab-page-head"><div><div class="ab-eyebrow">{html.escape(eyebrow)}</div>'
            f'<h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div></div>')


def welcome_hero_html() -> str:
    return (
        f'<section class="ab-hero">{MOTIF_SVG}<div class="ab-eyebrow">Explainable AI for Ayurveda</div>'
        "<h2>Traditional knowledge, clearly explained.</h2>"
        "<p>AyurBot answers questions about Ayurvedic herbs, doshas and classical terms using only its local "
        "knowledge base, and shows exactly which words it recognised and which entries it used.</p></section>"
    )


FEATURES = [
    ("💬", "Grounded answers", "Every sentence comes from the knowledge base. If nothing matches, AyurBot says so."),
    ("🔍", "Explainable NER", "Herbs, doshas, terms and dosage mentions are highlighted with how each was detected."),
    ("🧠", "Semantic search", "Sentence Transformers find relevant entries by meaning, with a BM25 fallback."),
    ("🌿", "Herb profiles", "Botanical names, rasa, virya, vipaka, dosha effects and safety notes."),
    ("⚖️", "Responsible by design", "Safety notices for dosage, personal and emergency questions."),
    ("📊", "Measured quality", "A built-in evaluation reports NER, retrieval accuracy and response time."),
]


def features_html() -> str:
    return '<div class="ab-feature-grid">' + "".join(
        f'<div class="ab-feature"><div class="ic">{icon}</div><h4>{html.escape(t)}</h4><p>{html.escape(d)}</p></div>'
        for icon, t, d in FEATURES) + "</div>"


def steps_html() -> str:
    steps = [("Validate", "The question is cleaned and checked."),
             ("Recognise", "Rule-based NER finds Ayurvedic entities."),
             ("Retrieve", "Semantic or lexical search finds matching entries."),
             ("Explain", "A grounded answer is built, with evidence and safety notes.")]
    return '<div class="ab-steps">' + "".join(
        f'<div class="ab-step"><b>{t}</b>{d}</div>' for t, d in steps) + "</div>"


def find_free_port(host: str, preferred: int, attempts: int) -> int | None:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host if host not in ("", "0.0.0.0") else "127.0.0.1", port))
                return port
            except OSError:
                continue
    return None


def make_handlers(service: AyurBotService):
    """Create the UI callbacks. Kept outside ``build_ui`` so tests can call them."""
    import gradio as gr
    import pandas as pd

    max_messages = service.settings.max_history_messages

    def queue_user(text: str, history: list):
        """Stage 1: show the question immediately and clear the box."""
        text = (text or "").strip()
        history = normalize_history(history, max_messages)
        if not text:
            return gr.skip(), history, ""
        history.append({"role": "user", "content": text})
        return "", history[-max_messages:], text

    def answer(pending: str, history: list):
        """Stage 2: compute the grounded answer and the explainability views."""
        history = normalize_history(history, max_messages)
        if not pending:
            return history, gr.skip(), gr.skip(), gr.skip(), gr.skip()
        result = service.ask(pending)
        history.append({"role": "assistant", "content": result.answer})
        return (
            history[-max_messages:],
            result.highlighted,
            result.entity_table(),
            result.evidence_table(),
            status_markdown(service, result),
        )

    def clear_all():
        return ([], [], pd.DataFrame(columns=ENTITY_COLUMNS), pd.DataFrame(columns=EVIDENCE_COLUMNS),
                status_markdown(service), "", "")

    return queue_user, answer, clear_all


# ---------------------------------------------------------------- screen helpers
def herb_names(service: AyurBotService) -> list[str]:
    return [h.canonical_name for h in service.kb.entries_of_type("herb")] if service.kb else []


def herb_detail(service: AyurBotService, name: str | None) -> str:
    if not service.kb or not service.engine:
        return "The knowledge base is not loaded. See the System screen."
    entry = service.kb.find_by_name(name or "")
    if entry is None or entry.type != "herb":
        return "Choose a herb to see its traditional profile."
    return service.engine._card(entry)


def herb_comparison(service: AyurBotService):
    import pandas as pd

    rows = []
    for h in service.kb.entries_of_type("herb") if service.kb else []:
        props = h.get("traditional_properties") or {}
        assoc = h.get("dosha_associations") or {}
        name = lambda ids: ", ".join(service.kb.get(i).canonical_name for i in ids if service.kb.get(i)) or "None recorded"
        rows.append({
            "Herb": h.canonical_name,
            "Botanical name": h.get("botanical_name") or "Not available",
            "English name": h.get("english_name") or "Not available",
            "Rasa (taste)": ", ".join(props.get("rasa", [])) or "Not available",
            "Virya": props.get("virya") or "Not available",
            "Vipaka": props.get("vipaka") or "Not available",
            "Pacifies": name(assoc.get("pacifies", [])),
            "May aggravate": name(assoc.get("may_aggravate", [])),
        })
    return pd.DataFrame(rows, columns=["Herb", "Botanical name", "English name", "Rasa (taste)", "Virya",
                                       "Vipaka", "Pacifies", "May aggravate"])


DOSHA_COLORS = {"vata": "var(--ab-vata)", "pitta": "var(--ab-pitta)", "kapha": "var(--ab-kapha)"}


def dosha_cards_html(service: AyurBotService) -> str:
    if not service.kb:
        return "<p>The knowledge base is not loaded.</p>"
    cards = []
    for d in service.kb.entries_of_type("dosha"):
        functions = "".join(f"<li>{html.escape(f)}</li>" for f in d.string_list("functions"))
        pac = ", ".join(h.canonical_name for h in service.kb.herbs_pacifying(d.id)) or "none recorded"
        agg = ", ".join(h.canonical_name for h in service.kb.herbs_aggravating(d.id)) or "none recorded"
        cards.append(
            f'<div class="ab-dosha-card" style="--c:{DOSHA_COLORS.get(d.id, "var(--ab-leaf)")}">'
            f"<h4>{html.escape(d.canonical_name)} {html.escape(' '.join(d.string_list('devanagari')))}</h4>"
            f'<div class="el">{html.escape(" + ".join(d.string_list("elements")))}</div>'
            f"<p>{html.escape(d.summary)}</p><ul>{functions}</ul>"
            f'<div class="herbs"><b>Traditionally said to pacify it:</b> {html.escape(pac)}<br>'
            f"<b>May aggravate it:</b> {html.escape(agg)}</div></div>"
        )
    note = ("<p style='margin-top:12px;font-size:.84rem;color:var(--ab-muted)'>Doshas are explanatory concepts of "
            "Ayurvedic theory, not measurable biomedical quantities. Herb associations are traditional descriptions "
            "from the demonstration knowledge base.</p>")
    return f'<div class="ab-dosha-grid">{"".join(cards)}</div>{note}'


def analyze_text(service: AyurBotService, text: str):
    """Run only the NER engine on a passage (no retrieval, no answer)."""
    import pandas as pd

    from src.ner_engine import highlight_spans
    from src.text_utils import clean_user_text

    text = clean_user_text(text)
    empty = pd.DataFrame(columns=ENTITY_COLUMNS)
    if not service.ner:
        return [], empty, "The knowledge base is not loaded, so entities cannot be detected."
    if not text:
        return [], empty, "Paste or type some text, then press **Analyse text**."
    text = text[:5000]
    entities = service.ner.extract(text)
    rows = [{"Text in query": e.text, "Category": e.label, "Canonical entry": e.canonical,
             "Offsets": f"{e.start}–{e.end}", "Detection method": e.method, "Evidence": e.evidence} for e in entities]
    counts: dict[str, int] = {}
    for e in entities:
        counts[e.label] = counts.get(e.label, 0) + 1
    if entities:
        summary = f"**{len(entities)} entities found:** " + ", ".join(f"{k} × {v}" for k, v in counts.items())
    else:
        summary = "No known Ayurvedic terms were found in this text."
    return highlight_spans(text, entities) or [(text, None)], pd.DataFrame(rows, columns=ENTITY_COLUMNS), summary


KB_TYPE_FILTERS = {"All types": None, "Herbs": "herb", "Doshas": "dosha", "Condition terms": "condition_term",
                   "Traditional properties": "traditional_property", "Concepts": "concept"}


def kb_table(service: AyurBotService, type_label: str = "All types", search: str = ""):
    import pandas as pd

    from src.text_utils import fold

    cols = ["Name", "Type", "Also known as", "Summary", "Data status"]
    if not service.kb:
        return pd.DataFrame(columns=cols)
    wanted = KB_TYPE_FILTERS.get(type_label)
    needle = fold(search or "").strip()
    rows = []
    for e in service.kb.entries:
        if wanted and e.type != wanted:
            continue
        aka = ", ".join(dict.fromkeys(e.string_list("synonyms") + e.string_list("approximate_english_terms")))
        haystack = fold(" ".join([e.canonical_name, aka, e.summary, e.get("botanical_name") or ""]))
        if needle and needle not in haystack:
            continue
        rows.append({"Name": e.canonical_name, "Type": e.type_display, "Also known as": aka or "-",
                     "Summary": e.summary, "Data status": e.get("data_status") or "unspecified"})
    return pd.DataFrame(rows, columns=cols)


def kb_stats_html(service: AyurBotService) -> str:
    counts = service.status.kb_counts
    items = [("Herbs", counts.get("herb", 0)), ("Doshas", counts.get("dosha", 0)),
             ("Condition terms", counts.get("condition_term", 0)),
             ("Properties", counts.get("traditional_property", 0)), ("Concepts", counts.get("concept", 0))]
    return '<div class="ab-stats">' + "".join(
        f'<div class="ab-stat"><b>{n}</b><span>{label}</span></div>' for label, n in items) + "</div>"


# ---------------------------------------------------------------- evaluation
# A small, hand-labelled demonstration test set. It checks the behaviour of this
# prototype; it is not a published benchmark and says nothing about medical accuracy.
EVAL_CASES = [
    {"q": "What is Tulsi?", "ents": [("HERB", "Tulsi")], "rel": ["tulsi"]},
    {"q": "Tell me about Holy Basil", "ents": [("HERB", "Tulsi")], "rel": ["tulsi"]},
    {"q": "Does Haridra pacify Kapha?", "ents": [("HERB", "Haridra"), ("DOSHA", "Kapha")], "rel": ["haridra"]},
    {"q": "Withania somnifera and Vata", "ents": [("BOTANICAL_NAME", "Ashwagandha"), ("DOSHA", "Vata")],
     "rel": ["ashwagandha"]},
    {"q": "Is Giloy described for Jwara?", "ents": [("HERB", "Guduchi"), ("CONDITION_TERM", "Jwara")],
     "rel": ["guduchi"]},
    {"q": "Neem for skin disease", "ents": [("HERB", "Neem"), ("CONDITION_TERM", "Kushtha")], "rel": ["neem"]},
    {"q": "What does Medhya mean?", "ents": [("TRADITIONAL_PROPERTY", "Medhya")], "rel": ["medhya"]},
    {"q": "What is Agni?", "ents": [("AYURVEDIC_CONCEPT", "Agni")], "rel": ["agni"]},
    {"q": "Explain Prakriti and Vikriti", "ents": [("AYURVEDIC_CONCEPT", "Prakriti"),
                                                   ("AYURVEDIC_CONCEPT", "Vikriti")], "rel": ["prakriti", "vikriti"]},
    {"q": "Which herbs are described for cough?", "ents": [("CONDITION_TERM", "Kasa")], "rel": ["kasa", "tulsi"]},
    {"q": "Is 5 g of turmeric too much?", "ents": [("DOSAGE_MENTION", "5 g"), ("HERB", "Haridra")],
     "rel": ["haridra"]},
    {"q": "तुलसी क्या है?", "ents": [("HERB", "Tulsi")], "rel": ["tulsi"]},
    {"q": "What is Pitta dosha?", "ents": [("DOSHA", "Pitta"), ("AYURVEDIC_CONCEPT", "Dosha")],
     "rel": ["pitta", "dosha"]},
    {"q": "Bacopa monnieri and memory", "ents": [("BOTANICAL_NAME", "Brahmi")], "rel": ["brahmi"]},
    {"q": "Ashwagandha twice a day", "ents": [("HERB", "Ashwagandha"), ("DOSAGE_MENTION", "twice a day")],
     "rel": ["ashwagandha"]},
    {"q": "What is Ama in Ayurveda?", "ents": [("AYURVEDIC_CONCEPT", "Ama")], "rel": ["ama"]},
    {"q": "Ushna virya herbs", "ents": [("TRADITIONAL_PROPERTY", "Ushna"), ("AYURVEDIC_CONCEPT", "Virya")],
     "rel": ["ushna", "virya"]},
    {"q": "Rasayana herbs like Guduchi", "ents": [("TRADITIONAL_PROPERTY", "Rasayana"), ("HERB", "Guduchi")],
     "rel": ["rasayana", "guduchi"]},
    {"q": "What is the weather in Pune today?", "ents": [], "rel": []},
    {"q": "Who won the cricket match yesterday?", "ents": [], "rel": []},
]

EVAL_CASE_COLUMNS = ["Query", "Expected entities", "Found entities", "NER", "Top retrieved", "Relevant rank",
                     "Answer focus", "Time (ms)"]
EVAL_LABEL_COLUMNS = ["Category", "Expected", "Found", "Correct", "Precision", "Recall", "F1"]


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def run_evaluation(service: AyurBotService):
    """Run the demonstration test set. Returns (metrics_html, label_df, case_df, notes_md)."""
    import pandas as pd

    from src.text_utils import fold

    empty = (pd.DataFrame(columns=EVAL_LABEL_COLUMNS), pd.DataFrame(columns=EVAL_CASE_COLUMNS))
    if not service.engine or not service.ner or not service.retrieval or not service.kb:
        return "<p>The knowledge base is not loaded, so the evaluation cannot run.</p>", *empty, ""

    norm = lambda pair: (pair[0], fold(" ".join(pair[1].split())))
    per_label: dict[str, list[int]] = {}
    tp = fp = fn = 0
    hits1 = hits3 = 0
    rr_total = 0.0
    answerable = rejections = rejections_ok = answer_ok = 0
    times: list[float] = []
    rows = []
    for case in EVAL_CASES:
        expected = {norm(e) for e in case["ents"]}
        found_ents = service.ner.extract(case["q"])
        found = {norm((e.label, e.canonical)) for e in found_ents}
        for label in {l for l, _ in expected | found}:
            exp_l = {x for x in expected if x[0] == label}
            got_l = {x for x in found if x[0] == label}
            counts = per_label.setdefault(label, [0, 0, 0])
            counts[0] += len(exp_l); counts[1] += len(got_l); counts[2] += len(exp_l & got_l)
        tp += len(expected & found); fp += len(found - expected); fn += len(expected - found)

        retrieved = [r.entry_id for r in service.retrieval.search(case["q"], top_k=3).results]
        response = service.ask(case["q"])
        times.append(response.elapsed_ms)
        rel = set(case["rel"])
        if rel:
            answerable += 1
            rank = next((i for i, rid in enumerate(retrieved, 1) if rid in rel), None)
            hits1 += rank == 1
            hits3 += rank is not None
            rr_total += 1 / rank if rank else 0.0
            focus = response.evidence_rows[0]["Entry"] if response.evidence_rows else "no answer"
            rel_names = {service.kb.get(r).canonical_name for r in rel if service.kb.get(r)}
            answer_ok += focus in rel_names
            rank_text = str(rank) if rank else "not in top 3"
        else:
            rejections += 1
            rejections_ok += response.status == "no_match"
            focus = "correctly declined" if response.status == "no_match" else "answered (should decline)"
            rank_text = "n/a"
        rows.append({
            "Query": case["q"],
            "Expected entities": ", ".join(f"{c} ({l})" for l, c in case["ents"]) or "none",
            "Found entities": ", ".join(f"{e.canonical} ({e.label})" for e in found_ents) or "none",
            "NER": "✓ exact" if expected == found else "✗ differs",
            "Top retrieved": ", ".join(service.kb.get(r).canonical_name for r in retrieved if service.kb.get(r)) or "none",
            "Relevant rank": rank_text,
            "Answer focus": focus,
            "Time (ms)": round(response.elapsed_ms, 1),
        })

    p, r, f = _prf(tp, fp, fn)
    label_rows = []
    for label in sorted(per_label, key=lambda l: ENTITY_LABELS.index(l) if l in ENTITY_LABELS else 99):
        exp_n, got_n, ok = per_label[label]
        lp, lr, lf = _prf(ok, got_n - ok, exp_n - ok)
        label_rows.append({"Category": label, "Expected": exp_n, "Found": got_n, "Correct": ok,
                           "Precision": f"{lp:.2f}", "Recall": f"{lr:.2f}", "F1": f"{lf:.2f}"})
    times_sorted = sorted(times)
    avg_ms = sum(times) / len(times)
    p95_ms = times_sorted[min(len(times_sorted) - 1, int(round(0.95 * (len(times_sorted) - 1))))]
    mode = service.refresh_status().retrieval_mode

    def metric(label: str, value: float, detail: str, pct: bool = True) -> str:
        shown = f"{value * 100:.0f}%" if pct else (f"{value:.1f} ms" if value < 10 else f"{value:.0f} ms")
        bar = f'<div class="ab-bar"><i style="width:{max(0, min(1, value)) * 100:.0f}%"></i></div>' if pct else ""
        return f'<div class="ab-metric"><span>{label}</span><b>{shown}</b><small>{detail}</small>{bar}</div>'

    cards = "".join([
        metric("NER F1 score", f, f"Precision {p:.2f}, recall {r:.2f}"),
        metric("Retrieval Hit@1", hits1 / answerable, f"Relevant entry ranked first ({hits1}/{answerable})"),
        metric("Retrieval Hit@3", hits3 / answerable, f"MRR {rr_total / answerable:.2f}"),
        metric("Answer focus accuracy", answer_ok / answerable, f"Main answer about the right entry ({answer_ok}/{answerable})"),
        metric("Off-topic rejection", rejections_ok / rejections if rejections else 1.0,
               f"Declined unrelated questions ({rejections_ok}/{rejections})"),
        metric("Average response time", avg_ms, f"95th percentile {p95_ms:.1f} ms", pct=False),
    ])
    notes = (
        f"Evaluated **{len(EVAL_CASES)} hand-labelled questions** using **{mode}** retrieval. "
        "NER is scored by exact category and canonical-entry match. Retrieval Hit@k and MRR use the retriever "
        "alone; *answer focus* checks the entry the final answer is mainly about. This is a small demonstration "
        "set for checking the prototype, not a benchmark, and it does not measure medical correctness."
    )
    return (f'<div class="ab-metric-grid">{cards}</div>', pd.DataFrame(label_rows, columns=EVAL_LABEL_COLUMNS),
            pd.DataFrame(rows, columns=EVAL_CASE_COLUMNS), notes)


def dosha_detail(service: AyurBotService, name: str | None) -> str:
    if not service.kb or not service.engine:
        return "The knowledge base is not loaded."
    entry = service.kb.find_by_name(name or "")
    if entry is None or entry.type != "dosha":
        return "Choose a dosha."
    pac = service.kb.herbs_pacifying(entry.id)
    agg = service.kb.herbs_aggravating(entry.id)
    pac_lines = "\n".join(f"- **{h.canonical_name}** (*{h.get('botanical_name')}*): {h.summary}" for h in pac)
    agg_lines = "\n".join(f"- **{h.canonical_name}**: {(h.get('dosha_associations') or {}).get('note', '')}"
                          for h in agg)
    return (f"{service.engine._card(entry)}\n\n**Herbs traditionally described as pacifying it**\n\n"
            f"{pac_lines or '- None recorded.'}\n\n**Herbs traditionally described as possibly aggravating it**"
            f"\n\n{agg_lines or '- None recorded.'}")


# ---------------------------------------------------------------- UI
def build_ui(service: AyurBotService):
    import gradio as gr
    import pandas as pd

    queue_user, answer, clear_all = make_handlers(service)
    empty_entities = pd.DataFrame(columns=ENTITY_COLUMNS)
    empty_evidence = pd.DataFrame(columns=EVIDENCE_COLUMNS)
    herbs = herb_names(service)
    doshas = [d.canonical_name for d in service.kb.entries_of_type("dosha")] if service.kb else []

    with gr.Blocks(title="AyurBot: Explainable Ayurveda assistant", analytics_enabled=False) as demo:
        pending = gr.State("")
        with gr.Row(equal_height=False):
            # ============================================================ sidebar
            with gr.Column(scale=0, min_width=250, elem_id="ab-sidebar"):
                gr.HTML(f'<div id="ab-brand">{LEAF_SVG}<div><h1>AyurBot</h1>'
                        "<p>Explainable Ayurveda assistant</p></div></div>")
                gr.Markdown("Navigate", elem_classes="ab-nav-label")
                nav_buttons = []
                with gr.Column(elem_id="ab-nav"):
                    for i, (_, icon, label) in enumerate(SCREENS):
                        nav_buttons.append(gr.Button(f"{icon}  {label}", variant="primary" if i == 0 else "secondary",
                                                     elem_classes="ab-nav", size="md"))
                side_status = gr.HTML(
                    f'<div id="ab-side-foot">{mode_pill_html(service)}'
                    f'<span class="ab-pill">Runs locally</span><br>Educational prototype. '
                    "Not medical advice.</div>")

            # ============================================================ main area
            with gr.Column(scale=1, min_width=340):
                screens = []

                # ---------------------------------------------------- 1. Welcome
                with gr.Column(visible=True, elem_classes="ab-screen") as s_welcome:
                    gr.HTML(welcome_hero_html())
                    with gr.Row(elem_id="ab-cta"):
                        cta_chat = gr.Button("💬  Start a conversation", variant="primary", scale=0, min_width=220)
                        cta_herbs = gr.Button("🌿  Explore herbs", variant="secondary", scale=0, min_width=180)
                        cta_eval = gr.Button("📊  See performance", variant="secondary", scale=0, min_width=180)
                    gr.HTML(kb_stats_html(service))
                    with gr.Column(elem_classes="ab-card"):
                        gr.Markdown("### What AyurBot can do", elem_classes="ab-section-title")
                        gr.HTML(features_html())
                    with gr.Column(elem_classes="ab-card"):
                        gr.Markdown("### How an answer is built", elem_classes="ab-section-title")
                        gr.HTML(steps_html())
                    gr.HTML(f'<div id="ab-disclaimer" role="note">{SHIELD_SVG}<span><b>For education only.</b> '
                            f'{html.escape(DISCLAIMER)}</span></div>')
                screens.append(s_welcome)

                # ---------------------------------------------------- 2. Ask AyurBot
                with gr.Column(visible=False, elem_classes="ab-screen") as s_chat:
                    gr.HTML(page_head("Conversation", "Ask AyurBot",
                                      "Ask about herbs, doshas or classical terms. Answers come only from the "
                                      "local knowledge base."))
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=7, min_width=340, elem_classes="ab-card"):
                            chatbot = gr.Chatbot(
                                show_label=False, height=500, render_markdown=True, sanitize_html=True,
                                group_consecutive_messages=False, elem_id="ab-chat",
                                placeholder="**Namaste 🙏 Ask about an herb, dosha or classical Ayurvedic term.**<br>"
                                            "Each answer shows its sources and how your words were recognised.",
                            )
                            with gr.Row(equal_height=True):
                                query = gr.Textbox(
                                    show_label=False, lines=1, max_lines=6, scale=8, elem_id="ab-input",
                                    max_length=service.settings.max_query_chars,
                                    placeholder="Ask a question, e.g. Does Tulsi pacify Kapha?  (Enter to send)",
                                )
                                send = gr.Button("Send", variant="primary", scale=1, min_width=90, elem_id="ab-send")
                            clear = gr.Button("Clear conversation", variant="secondary", size="sm")
                            gr.Markdown("Tap a question to ask it.", elem_classes="ab-topic-label")
                            chips: list[tuple[Any, str]] = []
                            for topic, questions in TOPICS.items():
                                gr.Markdown(topic, elem_classes="ab-topic-label")
                                with gr.Row():
                                    for q in questions:
                                        chips.append((gr.Button(q, size="sm", elem_classes="ab-chip", scale=0), q))
                        with gr.Column(scale=5, min_width=320, elem_classes="ab-card"):
                            gr.Markdown("### Explainability\nWhat was recognised and which entries were used.",
                                        elem_classes="ab-section-title")
                            highlighted = gr.HighlightedText(label="Recognised entities", color_map=ENTITY_COLORS,
                                                             show_legend=False, combine_adjacent=False,
                                                             show_inline_category=True)
                            gr.HTML(legend_html())
                            with gr.Tabs():
                                with gr.Tab("Entities"):
                                    entity_table = gr.Dataframe(value=empty_entities, show_label=False,
                                                                interactive=False, wrap=True, max_height=320)
                                with gr.Tab("Evidence"):
                                    evidence_table = gr.Dataframe(value=empty_evidence, show_label=False,
                                                                  interactive=False, wrap=True, max_height=320)
                                    gr.Markdown("<small>Scores are relevance indicators, not measures of medical "
                                                "accuracy.</small>")
                                with gr.Tab("Status"):
                                    status = gr.Markdown(status_markdown(service), elem_id="ab-status")
                screens.append(s_chat)

                # ---------------------------------------------------- 3. Herb Explorer
                with gr.Column(visible=False, elem_classes="ab-screen") as s_herbs:
                    gr.HTML(page_head("Materia medica", "Herb Explorer",
                                      "Traditional profiles of each herb in the knowledge base."))
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3, min_width=240, elem_classes="ab-card"):
                            gr.Markdown("### Choose a herb", elem_classes="ab-section-title")
                            herb_pick = gr.Radio(choices=herbs, value=herbs[0] if herbs else None, show_label=False)
                            ask_herb = gr.Button("Ask AyurBot about this herb", variant="primary", size="sm")
                        with gr.Column(scale=8, min_width=340, elem_classes="ab-card"):
                            herb_view = gr.Markdown(herb_detail(service, herbs[0] if herbs else None))
                    with gr.Column(elem_classes="ab-card"):
                        gr.Markdown("### Compare herbs\nKey traditional properties side by side.",
                                    elem_classes="ab-section-title")
                        gr.Dataframe(value=herb_comparison(service), show_label=False, interactive=False, wrap=True)
                screens.append(s_herbs)

                # ---------------------------------------------------- 4. Dosha Guide
                with gr.Column(visible=False, elem_classes="ab-screen") as s_doshas:
                    gr.HTML(page_head("Tridosha", "Dosha Guide",
                                      "Vata, Pitta and Kapha, their elements, functions and associated herbs."))
                    gr.HTML(dosha_cards_html(service))
                    with gr.Column(elem_classes="ab-card"):
                        gr.Markdown("### Explore a dosha", elem_classes="ab-section-title")
                        dosha_pick = gr.Radio(choices=doshas, value=doshas[0] if doshas else None, show_label=False)
                        dosha_view = gr.Markdown(dosha_detail(service, doshas[0] if doshas else None))
                screens.append(s_doshas)

                # ---------------------------------------------------- 5. Knowledge Base & System
                with gr.Column(visible=False, elem_classes="ab-screen") as s_kb:
                    gr.HTML(page_head("Data and system", "Knowledge Base & System",
                                      "Browse every entry, analyse any text with the NER engine, and check status."))
                    gr.HTML(kb_stats_html(service))
                    with gr.Tabs():
                        with gr.Tab("📚 Browse entries"):
                            with gr.Column(elem_classes="ab-card"):
                                with gr.Row():
                                    kb_type = gr.Dropdown(choices=list(KB_TYPE_FILTERS), value="All types",
                                                          label="Type", scale=1)
                                    kb_search = gr.Textbox(label="Search", placeholder="e.g. cough, Kapha, Giloy",
                                                           scale=3)
                                kb_view = gr.Dataframe(value=kb_table(service), show_label=False, interactive=False,
                                                       wrap=True, max_height=440)
                        with gr.Tab("🔍 Text analyser"):
                            with gr.Row(equal_height=False):
                                with gr.Column(scale=5, min_width=300, elem_classes="ab-card"):
                                    gr.Markdown("### Analyse any passage\nSee which Ayurvedic terms the rule-based "
                                                "NER engine recognises.", elem_classes="ab-section-title")
                                    passage = gr.Textbox(show_label=False, lines=8, max_lines=20,
                                                         placeholder="e.g. Haridra and Nimba are bitter herbs "
                                                                     "traditionally used for Kushtha and Kandu.")
                                    with gr.Row():
                                        analyse = gr.Button("Analyse text", variant="primary")
                                        reset = gr.Button("Clear", variant="secondary")
                                with gr.Column(scale=7, min_width=340, elem_classes="ab-card"):
                                    ner_summary = gr.Markdown("Paste or type some text, then press **Analyse text**.")
                                    ner_highlight = gr.HighlightedText(label="Highlighted entities",
                                                                       color_map=ENTITY_COLORS, show_legend=False,
                                                                       combine_adjacent=False,
                                                                       show_inline_category=True)
                                    gr.HTML(legend_html())
                                    ner_table = gr.Dataframe(value=empty_entities, label="Detection audit trail",
                                                             interactive=False, wrap=True, max_height=320)
                        with gr.Tab("⚙️ System status"):
                            with gr.Row(equal_height=False):
                                with gr.Column(elem_classes="ab-card"):
                                    gr.Markdown("### Status", elem_classes="ab-section-title")
                                    system_status = gr.Markdown(status_markdown(service), elem_id="ab-status")
                                    refresh = gr.Button("Refresh status", size="sm", variant="secondary")
                                with gr.Column(elem_classes="ab-card"):
                                    gr.Markdown(
                                        "### About AyurBot\n"
                                        "AyurBot combines **rule-based named entity recognition**, **semantic "
                                        "search** with Sentence Transformers (with a BM25 fallback) and "
                                        "**template-based answers** grounded in a local knowledge base. No "
                                        "generative AI model is used, and nothing leaves this computer.\n\n"
                                        "All entries are demonstration data summarising traditional Ayurvedic "
                                        "descriptions. They are not clinical evidence and have not been reviewed "
                                        "by a clinician.", elem_classes="ab-section-title")
                screens.append(s_kb)

                # ---------------------------------------------------- 6. Performance Evaluation
                with gr.Column(visible=False, elem_classes="ab-screen") as s_eval:
                    gr.HTML(page_head("Quality", "Performance Evaluation",
                                      f"Run a built-in set of {len(EVAL_CASES)} labelled questions to measure NER "
                                      "accuracy, retrieval quality and response time."))
                    with gr.Row():
                        run_eval = gr.Button("▶  Run evaluation", variant="primary", scale=0, min_width=200)
                    eval_cards = gr.HTML('<div class="ab-metric-grid"><div class="ab-metric"><span>Not run yet'
                                         "</span><b>–</b><small>Press Run evaluation to score the system.</small>"
                                         "</div></div>")
                    eval_notes = gr.Markdown("")
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5, min_width=320, elem_classes="ab-card"):
                            gr.Markdown("### NER by category", elem_classes="ab-section-title")
                            eval_labels = gr.Dataframe(value=pd.DataFrame(columns=EVAL_LABEL_COLUMNS),
                                                       show_label=False, interactive=False, wrap=True)
                        with gr.Column(scale=7, min_width=340, elem_classes="ab-card"):
                            gr.Markdown("### Metric definitions", elem_classes="ab-section-title")
                            gr.Markdown(
                                "- **Precision**: share of detected entities that were correct.\n"
                                "- **Recall**: share of expected entities that were detected.\n"
                                "- **F1**: balance of precision and recall.\n"
                                "- **Hit@k**: a relevant entry appears in the top *k* search results.\n"
                                "- **MRR**: mean reciprocal rank of the first relevant entry (1.0 is best).\n"
                                "- **Off-topic rejection**: unrelated questions are declined instead of guessed.")
                    with gr.Column(elem_classes="ab-card"):
                        gr.Markdown("### Per-question results", elem_classes="ab-section-title")
                        eval_cases = gr.Dataframe(value=pd.DataFrame(columns=EVAL_CASE_COLUMNS), show_label=False,
                                                  interactive=False, wrap=True, max_height=480)
                screens.append(s_eval)

        # ============================================================ navigation
        def go(index: int):
            return ([gr.update(visible=i == index) for i in range(len(screens))]
                    + [gr.update(variant="primary" if i == index else "secondary") for i in range(len(nav_buttons))])

        nav_outputs = screens + nav_buttons
        for i, button in enumerate(nav_buttons):
            button.click(lambda i=i: go(i), outputs=nav_outputs, queue=False, show_progress="hidden", api_name=False)
        index_of = {key: i for i, (key, _, _) in enumerate(SCREENS)}
        cta_chat.click(lambda: go(index_of["chat"]), outputs=nav_outputs, queue=False, api_name=False)
        cta_herbs.click(lambda: go(index_of["herbs"]), outputs=nav_outputs, queue=False, api_name=False)
        cta_eval.click(lambda: go(index_of["eval"]), outputs=nav_outputs, queue=False, api_name=False)

        # ============================================================ chat events
        outputs_stage2 = [chatbot, highlighted, entity_table, evidence_table, status]
        for trigger in (query.submit, send.click):
            trigger(queue_user, inputs=[query, chatbot], outputs=[query, chatbot, pending],
                    queue=False, show_progress="hidden").then(
                answer, inputs=[pending, chatbot], outputs=outputs_stage2, show_progress="minimal")
        for chip, q in chips:
            chip.click(lambda h, q=q: queue_user(q, h), inputs=[chatbot], outputs=[query, chatbot, pending],
                       queue=False, show_progress="hidden", api_name=False).then(
                answer, inputs=[pending, chatbot], outputs=outputs_stage2, show_progress="minimal",
                api_name=False)
        clear.click(clear_all, outputs=[chatbot, highlighted, entity_table, evidence_table, status, query, pending],
                    queue=False)
        ask_herb.click(lambda n: go(index_of["chat"]), inputs=herb_pick, outputs=nav_outputs,
                       queue=False, api_name=False).then(
            lambda n, h: queue_user(f"Tell me about {n}" if n else "", h), inputs=[herb_pick, chatbot],
            outputs=[query, chatbot, pending], queue=False, api_name=False).then(
            answer, inputs=[pending, chatbot], outputs=outputs_stage2, show_progress="minimal", api_name=False)

        # ============================================================ other screens
        herb_pick.change(lambda n: herb_detail(service, n), inputs=herb_pick, outputs=herb_view, api_name=False)
        dosha_pick.change(lambda n: dosha_detail(service, n), inputs=dosha_pick, outputs=dosha_view, api_name=False)
        for trigger in (analyse.click, passage.submit):
            trigger(lambda t: analyze_text(service, t), inputs=passage,
                    outputs=[ner_highlight, ner_table, ner_summary], api_name="analyse")
        reset.click(lambda: ("", [], empty_entities, "Paste or type some text, then press **Analyse text**."),
                    outputs=[passage, ner_highlight, ner_table, ner_summary], queue=False, api_name=False)
        for trigger in (kb_type.change, kb_search.change):
            trigger(lambda t, q: kb_table(service, t, q), inputs=[kb_type, kb_search], outputs=kb_view,
                    queue=False, api_name=False)
        refresh.click(lambda: (status_markdown(service), f'<div id="ab-side-foot">{mode_pill_html(service)}'
                               '<span class="ab-pill">Runs locally</span><br>Educational prototype. '
                               "Not medical advice.</div>"),
                      outputs=[system_status, side_status], queue=False, api_name=False)
        run_eval.click(lambda: run_evaluation(service), outputs=[eval_cards, eval_labels, eval_cases, eval_notes],
                       show_progress="minimal", api_name="evaluate")
    return demo


def theme():
    import gradio as gr

    sage = gr.themes.Color(
        c50="#f1f5ef", c100="#e9f0e6", c200="#d6e2d3", c300="#b6cbb3", c400="#93b094",
        c500="#7a9a7e", c600="#5f8066", c700="#4b6b55", c800="#3b5444", c900="#2f4236", c950="#1f2d24",
    )
    return gr.themes.Soft(
        primary_hue=sage,
        secondary_hue="amber",
        neutral_hue="stone",
        radius_size="md",
        font=["Segoe UI", "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    ).set(
        body_background_fill="#f6f1e6",
        background_fill_primary="#fffcf6",
        background_fill_secondary="#f3eddf",
        block_background_fill="#fffcf6",
        block_border_width="0px",
        block_shadow="none",
        border_color_primary="#e5ddcb",
        input_background_fill="#fffefb",
        button_primary_background_fill="#4b6b55",
        button_primary_background_fill_hover="#3b5444",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#fffcf6",
        button_secondary_background_fill_hover="#e9f0e6",
        button_secondary_text_color="#2f4236",
        button_large_radius="12px",
        input_radius="12px",
    )


# ---------------------------------------------------------------- main
def configure_logging() -> None:
    try:  # Windows consoles may not be UTF-8; never crash on printing.
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "sentence_transformers", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(settings: Settings | None = None) -> int:
    configure_logging()
    settings = settings or get_settings()
    for warning in settings.warnings:
        log.warning("Config: %s", warning)
    log.info("Starting AyurBot...")
    service = AyurBotService(settings)
    st = service.status
    if st.kb_ok:
        log.info("Knowledge base: %s", st.kb_message)
        log.info("Retrieval mode: %s", service.mode_label)
    else:
        log.error("Starting in error mode; fix the knowledge base and restart.")

    port = find_free_port(settings.host, settings.port, settings.port_search_range)
    if port is None:
        log.error("Ports %d-%d are all in use. Set AYURBOT_PORT to a free port, e.g. AYURBOT_PORT=7900.",
                  settings.port, settings.port + settings.port_search_range - 1)
        return 1
    if port != settings.port:
        log.warning("Port %d is busy; using %d instead.", settings.port, port)

    demo = build_ui(service)
    demo.queue(default_concurrency_limit=settings.concurrency_limit, max_size=64)
    url_host = "127.0.0.1" if settings.host in ("0.0.0.0", "") else settings.host
    print(f"\n  AyurBot is ready:  http://{url_host}:{port}\n  Press Ctrl+C to stop.\n", flush=True)
    try:
        demo.launch(
            server_name=settings.host,
            server_port=port,
            inbrowser=settings.open_browser,
            show_error=True,
            quiet=True,
            ssr_mode=False,
            theme=theme(),
            css=CSS,
            js=FORCE_LIGHT_JS,
        )
    except KeyboardInterrupt:
        log.info("Stopped.")
    except OSError as exc:
        log.error("Could not start the web server on port %d: %s. Try AYURBOT_PORT=7900.", port, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
    