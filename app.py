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
EXAMPLES = [
    "What is Tulsi?",
    "Does Haridra pacify Kapha?",
    "Which herbs are traditionally described for cough?",
    "Tell me about Withania somnifera and Vata",
    "What does Medhya mean?",
    "Compare Neem and Guduchi for skin disease",
    "What is the difference between Prakriti and Vikriti?",
    "तुलसी क्या है?",
]

CSS = """
:root {
  --ab-green: #2e6b47; --ab-green-dark: #1f4d33; --ab-mint: #f1f8f3; --ab-mint-line: #d6e9dc;
  --ab-turmeric: #b7790c; --ab-turmeric-soft: #fff8ea; --ab-line: #e6ebe7;
  --ab-ink: #17241c; --ab-muted: #627068;
}
body, .gradio-container { background: #ffffff !important; }
.gradio-container { max-width: 1360px !important; margin: 0 auto !important; padding: 0 24px 28px !important; }
footer { display: none !important; }

/* Header */
#ab-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  padding: 20px 2px 16px; border-bottom: 1px solid var(--ab-line); margin-bottom: 16px; }
#ab-brand { display: flex; align-items: center; gap: 14px; }
#ab-brand h1 { font-family: Georgia, "Times New Roman", serif; font-size: 1.75rem; font-weight: 600;
  margin: 0; color: var(--ab-green-dark); letter-spacing: -0.01em; line-height: 1.1; }
#ab-brand p { margin: 4px 0 0; font-size: 0.92rem; color: var(--ab-muted); }
#ab-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.ab-pill { display: inline-flex; align-items: center; gap: 7px; font-size: 0.8rem; font-weight: 600;
  padding: 6px 12px; border-radius: 999px; background: var(--ab-mint); color: var(--ab-green-dark);
  border: 1px solid var(--ab-mint-line); }
.ab-dot { width: 8px; height: 8px; border-radius: 50%; background: #2fa35a; }
.ab-dot.warn { background: #d99a15; }
.ab-dot.err { background: #d4483b; }

/* Safety notice */
#ab-disclaimer { display: flex; gap: 12px; align-items: flex-start; background: var(--ab-turmeric-soft);
  border: 1px solid #f1dfb5; border-radius: 12px; padding: 12px 16px; color: #4a3a12;
  font-size: 0.88rem; line-height: 1.55; margin-bottom: 8px; }
#ab-disclaimer svg { flex: none; margin-top: 1px; }
#ab-disclaimer b { color: #7a5108; }

/* Cards */
.ab-card { background: #ffffff !important; border: 1px solid var(--ab-line) !important; border-radius: 16px !important;
  padding: 18px !important; box-shadow: 0 1px 3px rgba(23,36,28,.04), 0 6px 18px rgba(23,36,28,.04); }
.ab-section-title h3 { margin: 0 0 3px !important; font-size: 1.05rem !important; color: var(--ab-ink); font-weight: 650; }
.ab-section-title p { margin: 0 !important; color: var(--ab-muted); font-size: 0.86rem; }
#ab-chat { border: 1px solid var(--ab-line) !important; border-radius: 12px !important; background: #fbfdfb !important; }
#ab-input textarea { font-size: 0.98rem !important; background: #ffffff !important; }
#ab-send { min-height: 46px; align-self: stretch; }

/* Quick topics */
.ab-topic-label p { margin: 10px 0 4px !important; font-size: 0.78rem; font-weight: 650; color: var(--ab-muted); }
.ab-chip { border-radius: 999px !important; background: #ffffff !important; border: 1px solid var(--ab-mint-line) !important;
  color: var(--ab-green-dark) !important; font-weight: 500 !important; font-size: 0.84rem !important;
  box-shadow: none !important; }
.ab-chip:hover { background: var(--ab-mint) !important; border-color: var(--ab-green) !important; }
.ab-chip:focus-visible { outline: 2px solid var(--ab-green) !important; outline-offset: 2px; }

/* Explainability */
#ab-legend { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 6px 14px;
  font-size: 0.8rem; color: var(--ab-muted); padding: 10px 12px; background: #fafcfa;
  border: 1px solid var(--ab-line); border-radius: 10px; margin: 6px 0 4px; }
#ab-legend b { color: var(--ab-ink); font-weight: 600; margin-right: 4px; font-size: 0.74rem; }
#ab-legend span.sw { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 6px;
  border: 1px solid rgba(0,0,0,.15); }
#ab-status { font-size: 0.86rem; line-height: 1.55; }
#ab-footer { text-align: center; color: var(--ab-muted); font-size: 0.8rem; padding-top: 14px; }
@media (max-width: 720px) {
  .gradio-container { padding: 0 10px 18px !important; }
  #ab-brand h1 { font-size: 1.45rem; }
  .ab-card { padding: 12px !important; }
}
"""
# Always open in light mode so the page stays white.
FORCE_LIGHT_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""

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
    '<svg width="42" height="42" viewBox="0 0 48 48" aria-hidden="true">'
    '<rect width="48" height="48" rx="12" fill="#2e6b47"/>'
    '<path d="M14 34c0-12 8-20 22-21-1 14-9 22-21 22" fill="#9fd8a9"/>'
    '<path d="M14 35c5-7 10-12 17-17" stroke="#1f3d2b" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
    '</svg>'
)


SHIELD_SVG = (
    '<svg width="20" height="20" viewBox="0 0 24 24" aria-hidden="true"><path d="M12 2 4 5v6c0 5 3.4 9.4 8 11 '
    '4.6-1.6 8-6 8-11V5l-8-3z" fill="#b7790c"/><path d="M11 7h2v6h-2zM11 15h2v2h-2z" fill="#fff"/></svg>'
)


def header_html(service: AyurBotService) -> str:
    st = service.refresh_status()
    if not st.kb_ok:
        mode_pill = '<span class="ab-pill"><span class="ab-dot err"></span>Knowledge base error</span>'
    elif st.retrieval_mode == "semantic":
        mode_pill = '<span class="ab-pill"><span class="ab-dot"></span>Semantic search active</span>'
    else:
        mode_pill = '<span class="ab-pill"><span class="ab-dot warn"></span>Lexical fallback mode</span>'
    entries = sum(st.kb_counts.values())
    return (
        '<header id="ab-header"><div id="ab-brand">' + LEAF_SVG +
        '<div><h1>AyurBot</h1><p>Explainable Ayurveda knowledge assistant for herbs, doshas and classical terms</p></div></div>'
        f'<div id="ab-pills">{mode_pill}'
        f'<span class="ab-pill">{entries} knowledge entries</span>'
        '<span class="ab-pill">Private: runs on this computer</span></div></header>'
    )


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


# ---------------------------------------------------------------- UI
def build_ui(service: AyurBotService):
    import gradio as gr
    import pandas as pd

    queue_user, answer, clear_all = make_handlers(service)
    empty_entities = pd.DataFrame(columns=ENTITY_COLUMNS)
    empty_evidence = pd.DataFrame(columns=EVIDENCE_COLUMNS)

    with gr.Blocks(title="AyurBot: Explainable Ayurveda assistant", analytics_enabled=False) as demo:
        gr.HTML(header_html(service))
        gr.HTML(f'<div id="ab-disclaimer" role="note">{SHIELD_SVG}<span><b>For education only.</b> '
                f'{html.escape(DISCLAIMER)}</span></div>')

        pending = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=7, min_width=360, elem_classes="ab-card"):
                gr.Markdown("### Conversation\nAnswers come only from the local knowledge base.",
                            elem_classes="ab-section-title")
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=520,
                    placeholder="**Namaste. Ask about an herb, dosha or classical Ayurvedic term.**<br>"
                                "Each answer shows its sources and how your words were recognised.",
                    render_markdown=True,
                    sanitize_html=True,
                    group_consecutive_messages=False,
                    elem_id="ab-chat",
                )
                with gr.Row(elem_id="ab-actions", equal_height=True):
                    query = gr.Textbox(
                        show_label=False,
                        placeholder="Ask a question, e.g. Does Tulsi pacify Kapha?  (Enter to send, Shift+Enter for a new line)",
                        lines=1,
                        max_lines=6,
                        max_length=service.settings.max_query_chars,
                        autofocus=True,
                        scale=8,
                        elem_id="ab-input",
                    )
                    send = gr.Button("Send", variant="primary", scale=1, min_width=90, elem_id="ab-send")
                with gr.Row():
                    clear = gr.Button("Clear conversation", variant="secondary", size="sm", scale=0, min_width=170)
                gr.Markdown("Not sure where to start? Tap a question to ask it.", elem_classes="ab-topic-label")
                chips: list[tuple[Any, str]] = []
                for topic, questions in TOPICS.items():
                    gr.Markdown(topic, elem_classes="ab-topic-label")
                    with gr.Row():
                        for q in questions:
                            chips.append((gr.Button(q, size="sm", elem_classes="ab-chip", scale=0), q))

            with gr.Column(scale=5, min_width=330, elem_classes="ab-card"):
                gr.Markdown("### Explainability\nWhat was recognised in your question and which entries were used.",
                            elem_classes="ab-section-title")
                highlighted = gr.HighlightedText(
                    label="Recognised entities",
                    color_map=ENTITY_COLORS,
                    show_legend=False,
                    combine_adjacent=False,
                    show_inline_category=True,
                )
                gr.HTML(legend_html())
                with gr.Tabs():
                    with gr.Tab("Entities"):
                        entity_table = gr.Dataframe(
                            value=empty_entities, show_label=False,
                            interactive=False, wrap=True, max_height=340,
                        )
                    with gr.Tab("Evidence"):
                        evidence_table = gr.Dataframe(
                            value=empty_evidence, show_label=False,
                            interactive=False, wrap=True, max_height=340,
                        )
                        gr.Markdown(
                            "<small>Scores show how closely an entry matches the wording or meaning of your question. "
                            "They are relevance indicators, not measures of medical accuracy.</small>"
                        )
                    with gr.Tab("System status"):
                        status = gr.Markdown(status_markdown(service), elem_id="ab-status")

        gr.HTML('<div id="ab-footer">AyurBot 2.0. Rule-based NER, Sentence Transformers retrieval and a local, '
                'editable knowledge base. Not a substitute for professional medical advice.</div>')

        outputs_stage2 = [chatbot, highlighted, entity_table, evidence_table, status]
        for trigger in (query.submit, send.click):
            trigger(
                queue_user, inputs=[query, chatbot], outputs=[query, chatbot, pending],
                queue=False, show_progress="hidden",
            ).then(
                answer, inputs=[pending, chatbot], outputs=outputs_stage2,
                show_progress="minimal",
            )
        for chip, q in chips:
            chip.click(
                lambda h, q=q: queue_user(q, h), inputs=[chatbot], outputs=[query, chatbot, pending],
                queue=False, show_progress="hidden", api_name=False,
            ).then(answer, inputs=[pending, chatbot], outputs=outputs_stage2, show_progress="minimal",
                   api_name=False)
        clear.click(
            clear_all,
            outputs=[chatbot, highlighted, entity_table, evidence_table, status, query, pending],
            queue=False,
        )
    return demo


def theme():
    import gradio as gr

    leaf = gr.themes.Color(
        c50="#eef6f0", c100="#d9eadf", c200="#b5d6c0", c300="#88bb98", c400="#5c9c72",
        c500="#3d8158", c600="#2f6b45", c700="#275839", c800="#1f4630", c900="#1a3a28", c950="#0f241a",
    )
    return gr.themes.Soft(
        primary_hue=leaf,
        secondary_hue="amber",
        neutral_hue="slate",
        radius_size="md",
        font=["Segoe UI", "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    ).set(
        body_background_fill="#ffffff",
        background_fill_primary="#ffffff",
        background_fill_secondary="#f7faf8",
        block_background_fill="#ffffff",
        block_border_width="0px",
        block_shadow="none",
        button_primary_background_fill="#2f6b45",
        button_primary_background_fill_hover="#275839",
        button_primary_text_color="#ffffff",
        button_large_radius="10px",
        input_radius="10px",
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
    