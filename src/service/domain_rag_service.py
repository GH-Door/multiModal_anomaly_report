from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DomainKnowledgeRagService:
    """Retrieve domain knowledge context and build report prompt for API LLM flow."""

    def __init__(
        self,
        *,
        json_path: str,
        persist_dir: str,
        enabled: bool = True,
        top_k: int = 3,
    ) -> None:
        self.enabled = bool(enabled)
        self.top_k = max(1, int(top_k))
        self.json_path = Path(json_path)
        self.persist_dir = Path(persist_dir)

        self._retriever = None
        self._report_prompt_rag = None
        self._ready = False

        self._initialize()

    @property
    def ready(self) -> bool:
        return self.enabled and self._ready and self._retriever is not None and self._report_prompt_rag is not None

    def _initialize(self) -> None:
        if not self.enabled:
            logger.info("Domain knowledge RAG disabled by config")
            return

        if not self.json_path.exists():
            logger.warning("Domain knowledge JSON not found: %s", self.json_path)
            return

        try:
            from src.rag import Indexer, Retrievers
            from src.rag.prompt import report_prompt_rag

            indexer = Indexer(json_path=str(self.json_path), persist_dir=str(self.persist_dir))
            vectorstore = indexer.get_or_create()
            self._retriever = Retrievers(vectorstore)
            self._report_prompt_rag = report_prompt_rag
            self._ready = True

            count = "unknown"
            try:
                count = str(vectorstore._collection.count())  # type: ignore[attr-defined]
            except Exception:
                pass
            logger.info(
                "Domain knowledge RAG ready | docs=%s | json=%s | persist=%s",
                count,
                self.json_path,
                self.persist_dir,
            )
        except Exception:
            logger.exception(
                "Failed to initialize domain knowledge RAG | json=%s persist=%s",
                self.json_path,
                self.persist_dir,
            )

    @staticmethod
    def _split_model_category(model_category: str) -> tuple[str | None, str]:
        raw = str(model_category or "").strip().strip("/")
        if not raw:
            return None, ""
        if "/" not in raw:
            return None, raw
        dataset, category = raw.split("/", 1)
        return (dataset or None), (category or raw)

    @staticmethod
    def _build_query(category: str, ad_data: dict[str, Any] | None) -> str:
        if not category:
            return "defect anomaly inspection"

        ad_decision = str((ad_data or {}).get("ad_decision", "")).lower()
        if ad_decision == "normal":
            return f"{category} normal product appearance"

        region = (ad_data or {}).get("region")
        area_ratio = (ad_data or {}).get("area_ratio")
        extra: list[str] = []
        if isinstance(region, str) and region.strip():
            extra.append(f"region {region.strip()}")
        if area_ratio is not None:
            extra.append(f"area_ratio {area_ratio}")

        suffix = f" {' '.join(extra)}" if extra else ""
        return f"{category} defect anomaly inspection{suffix}"

    def build_report_instruction(
        self,
        *,
        model_category: str,
        ad_data: dict[str, Any] | None = None,
        ad_info_text: str = "",
    ) -> str | None:
        """Return report prompt with retrieved domain knowledge, or None when unavailable."""
        if not self.ready:
            return None

        dataset, category = self._split_model_category(model_category)
        if not category:
            return None

        query = self._build_query(category, ad_data)

        try:
            docs = self._retriever.retrieve(  # type: ignore[union-attr]
                query,
                dataset=dataset,
                category=category,
                defect_type=None,
                k=self.top_k,
            )
            if not docs and dataset:
                docs = self._retriever.retrieve(  # type: ignore[union-attr]
                    query,
                    dataset=None,
                    category=category,
                    defect_type=None,
                    k=self.top_k,
                )
            context = self._retriever.format_context(docs)  # type: ignore[union-attr]
            logger.info(
                "Domain knowledge RAG retrieved %d docs | dataset=%s | category=%s",
                len(docs),
                dataset or "*",
                category,
            )
            return self._report_prompt_rag(  # type: ignore[misc]
                category=category,
                domain_knowledge=context,
                ad_info=ad_info_text,
            )
        except Exception:
            logger.exception("Domain knowledge retrieval failed | model_category=%s", model_category)
            return None
