from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _resolve_few_shot_path(ref_path: str) -> str | None:
    candidate = Path(ref_path)
    if candidate.exists():
        return str(candidate)

    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        return None

    data_root = Path(data_dir)
    prefixes = (
        "/home/ubuntu/dataset/",
    )
    for prefix in prefixes:
        if not ref_path.startswith(prefix):
            continue
        suffix = ref_path[len(prefix) :]
        mapped = data_root / suffix
        if mapped.exists():
            return str(mapped)
    return None


class LlmService:
    """LLM report adapter returning DB-compatible keys."""

    def __init__(self, client, domain_rag_service=None) -> None:
        self.client = client
        self.domain_rag_service = domain_rag_service

    def generate_dynamic_report(
        self,
        image_path: str,
        ref_path: str | None,
        category: str,
        ad_data: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> Dict[str, Any]:
        del policy  # reserved for future prompt control
        ad_decision_raw = str(ad_data.get("ad_decision", "review_needed"))
        few_shot_paths: list[str] = []
        if ref_path:
            resolved = _resolve_few_shot_path(ref_path)
            if resolved:
                few_shot_paths = [resolved]
            else:
                logger.warning("Skipping unavailable RAG reference image for LLM: %s", ref_path)

        ad_info = {
            "anomaly_score": ad_data.get("ad_score", 0.0),
            "is_anomaly": ad_decision_raw.lower() == "anomaly",
            "defect_location": {
                "region": ad_data.get("region"),
                "area_ratio": ad_data.get("area_ratio"),
            },
        }

        report_instruction: str | None = None
        if self.domain_rag_service is not None:
            try:
                report_instruction = self.domain_rag_service.build_report_instruction(
                    model_category=category,
                    ad_data=ad_data,
                )
            except Exception:
                logger.exception("Domain knowledge RAG prompt build failed for category=%s", category)

        result = self.client.generate_report(
            image_path=image_path,
            category=category,
            ad_info=ad_info,
            few_shot_paths=few_shot_paths,
            instruction=report_instruction,
        )
        return {
            "is_anomaly_llm": result.get("is_anomaly_LLM"),
            "llm_report": result.get("llm_report"),
            "llm_summary": result.get("llm_summary"),
            "llm_inference_duration": result.get("llm_inference_duration"),
            "ad_decision": ad_decision_raw.lower(),
        }
