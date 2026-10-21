"""Builds every AyurBot component exactly once and exposes a single ``ask``.

Startup never raises. If the knowledge base is broken, the service starts in
an error state and every answer explains what to fix. If the transformer model
cannot be loaded, the service runs in lexical fallback mode.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from config import Settings

from .knowledge_base import KnowledgeBase, KnowledgeBaseError
from .ner_engine import NEREngine
from .response_engine import ChatResponse, ResponseEngine
from .retrieval_engine import (
    LEXICAL,
    SEMANTIC,
    LexicalRetriever,
    ModelFactory,
    RetrievalEngine,
    SemanticRetriever,
    default_model_factory,
)

log = logging.getLogger(__name__)


@dataclass
class ServiceStatus:
    kb_ok: bool = False
    kb_message: str = ""
    kb_counts: dict[str, int] = field(default_factory=dict)
    kb_warnings: list[str] = field(default_factory=list)
    ner_terms: int = 0
    retrieval_mode: str = LEXICAL
    model_state: str = "not_started"
    model_detail: str = ""
    startup_seconds: float = 0.0
    config_warnings: list[str] = field(default_factory=list)


class AyurBotService:
    def __init__(self, settings: Settings, model_factory: ModelFactory | None = None):
        self.settings = settings
        self.status = ServiceStatus(config_warnings=list(settings.warnings))
        self.kb: KnowledgeBase | None = None
        self.ner: NEREngine | None = None
        self.retrieval: RetrievalEngine | None = None
        self.engine: ResponseEngine | None = None
        started = time.perf_counter()
        self._start(model_factory)
        self.status.startup_seconds = round(time.perf_counter() - started, 2)

    def _start(self, model_factory: ModelFactory | None) -> None:
        s = self.settings
        try:
            self.kb = KnowledgeBase.from_file(s.kb_path)
        except KnowledgeBaseError as exc:
            self.status.kb_message = str(exc)
            log.error("Knowledge base could not be loaded:\n%s", exc)
            return
        self.status.kb_ok = True
        self.status.kb_counts = self.kb.counts()
        self.status.kb_warnings = list(self.kb.warnings)
        self.status.kb_message = f"Loaded {len(self.kb.entries)} entries from {s.kb_path.name}."
        for warning in self.kb.warnings:
            log.warning("Knowledge base: %s", warning)

        self.ner = NEREngine(self.kb)
        self.status.ner_terms = self.ner.term_count
        documents = self.kb.documents
        lexical = LexicalRetriever(documents)

        semantic = SemanticRetriever(
            documents,
            model_factory or default_model_factory(s.model_name, s.model_cache_dir, s.device, s.offline),
            model_name=s.model_name,
        )
        if s.disable_transformer:
            semantic.disable("Disabled by AYURBOT_DISABLE_TRANSFORMER.")
        else:
            log.info("Loading semantic model '%s' (first run may download about 90 MB)...", s.model_name)
            semantic.initialize()

        self.retrieval = RetrievalEngine(
            lexical=lexical,
            semantic=semantic,
            top_k=s.top_k,
            min_semantic_score=s.min_semantic_score,
            min_lexical_score=s.min_lexical_score,
        )
        self.engine = ResponseEngine(self.kb, self.ner, self.retrieval,
                                     max_query_chars=s.max_query_chars, max_cards=s.top_k)
        self.refresh_status()

    def refresh_status(self) -> ServiceStatus:
        if self.retrieval and self.retrieval.semantic:
            sem = self.retrieval.semantic.status
            self.status.model_state = sem.state
            self.status.model_detail = sem.detail
            self.status.retrieval_mode = self.retrieval.mode
        return self.status

    def ask(self, query: object) -> ChatResponse:
        if self.engine is None:
            return ChatResponse(
                query=str(query or ""),
                answer=("AyurBot cannot answer because the knowledge base failed to load.\n\n"
                        f"```\n{self.status.kb_message}\n```\n\nFix the file and restart the app."),
                status="error",
            )
        try:
            return self.engine.respond(query)
        except Exception as exc:  # last-resort guard: report, never crash the UI
            log.exception("Unexpected error while answering")
            return ChatResponse(
                query=str(query or ""),
                answer=f"Something went wrong while answering ({exc.__class__.__name__}: {exc}). "
                       "The error was logged in the terminal. Please try rephrasing the question.",
                status="error",
            )
        finally:
            self.refresh_status()

    @property
    def mode_label(self) -> str:
        mode = self.refresh_status().retrieval_mode
        return "Semantic search (Sentence Transformers)" if mode == SEMANTIC else "Lexical search (BM25 fallback)"
