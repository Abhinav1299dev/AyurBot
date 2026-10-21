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
:root { --ab-leaf: #2f5d3a; --ab-ink: #1d2a21; --ab-turmeric: #c98a12; --ab-sage: #eef3ec; --ab-muted: #56665a; }
.dark { --ab-leaf: #8cc79a; --ab-ink: #e6efe7; --ab-sage: #1c2620; --ab-muted: #a8b8ab; }
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; }
#ab-header { padding: 18px 4px 6px; }
#ab-header h1 { font-family: Georgia, "Times New Roman", serif; font-weight: 600; font-size: 2.1rem;
  letter-spacing: -0.01em; color: var(--ab-leaf); margin: 0 0 4px; }
#ab-header p { color: var(--ab-muted); margin: 0; max-width: 72ch; line-height: 1.5; }
#ab-disclaimer { border-left: 4px solid var(--ab-turmeric); background: var(--ab-sage);
  padding: 10px 14px; border-radius: 4px; color: var(--ab-ink); font-size: 0.92rem; line-height: 1.45; }
#ab-legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 0.84rem; color: var(--ab-ink); }
#ab-legend span.sw { display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 6px;
  vertical-align: -1px; border: 1px solid rgba(0,0,0,.25); }
#ab-legend b { font-weight: 600; }
#ab-status { font-size: 0.88rem; }
.ab-panel-title { margin: 2px 0 0 !important; color: var(--ab-ink); }
"""


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
        gr.HTML(
            '<header id="ab-header"><h1>AyurBot</h1>'
            "<p>Ask about Ayurvedic herbs, doshas and classical terms. Every answer is built only from the "
            "local knowledge base and shows which words were recognised and which entries were used.</p></header>"
        )
        gr.HTML(f'<div id="ab-disclaimer" role="note"><b>Educational prototype.</b> {html.escape(DISCLAIMER)}</div>')

        pending = gr.State("")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=340):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=540,
                    placeholder="Ask a question below, or pick an example.",
                    render_markdown=True,
                    sanitize_html=True,
                    group_consecutive_messages=False,
                )
                query = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Does Tulsi pacify Kapha?  (Enter to send, Shift+Enter for a new line)",
                    lines=2,
                    max_lines=6,
                    max_length=service.settings.max_query_chars,
                    autofocus=True,
                )
                with gr.Row():
                    send = gr.Button("Send question", variant="primary", scale=3)
                    clear = gr.Button("Clear chat", variant="secondary", scale=1)
                gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[query], label="Example questions")

            with gr.Column(scale=2, min_width=320):
                gr.Markdown("### How this answer was built", elem_classes="ab-panel-title")
                highlighted = gr.HighlightedText(
                    label="Recognised entities in your question",
                    color_map=ENTITY_COLORS,
                    show_legend=False,
                    combine_adjacent=False,
                    show_inline_category=True,
                )
                gr.HTML(legend_html())
                with gr.Tabs():
                    with gr.Tab("Entities"):
                        entity_table = gr.Dataframe(
                            value=empty_entities, label="Rule-based NER audit trail",
                            interactive=False, wrap=True, max_height=360,
                        )
                    with gr.Tab("Evidence"):
                        evidence_table = gr.Dataframe(
                            value=empty_evidence, label="Knowledge-base entries used",
                            interactive=False, wrap=True, max_height=360,
                        )
                        gr.Markdown(
                            "Scores show how closely an entry matches the wording or meaning of your question. "
                            "They are relevance indicators, not measures of medical accuracy or certainty."
                        )
                    with gr.Tab("System status"):
                        status = gr.Markdown(status_markdown(service), elem_id="ab-status")

        outputs_stage2 = [chatbot, highlighted, entity_table, evidence_table, status]
        for trigger in (query.submit, send.click):
            trigger(
                queue_user, inputs=[query, chatbot], outputs=[query, chatbot, pending],
                queue=False, show_progress="hidden",
            ).then(
                answer, inputs=[pending, chatbot], outputs=outputs_stage2,
                show_progress="minimal",
            )
        clear.click(
            clear_all,
            outputs=[chatbot, highlighted, entity_table, evidence_table, status, query, pending],
            queue=False,
        )
    return demo


def theme():
    import gradio as gr

    return gr.themes.Soft(
        primary_hue="green",
        secondary_hue="amber",
        neutral_hue="stone",
        font=["Segoe UI", "system-ui", "-apple-system", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
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
        )
    except KeyboardInterrupt:
        log.info("Stopped.")
    except OSError as exc:
        log.error("Could not start the web server on port %d: %s. Try AYURBOT_PORT=7900.", port, exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
