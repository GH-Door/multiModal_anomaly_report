from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from src.mllm.base import format_ad_info

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
            "decision": ad_decision_raw.lower(),
            "is_anomaly": ad_decision_raw.lower() == "anomaly",
            "decision_confidence": ad_data.get("confidence"),
            "reason_codes": ad_data.get("reason_codes"),
            "defect_location": {
                "has_defect": ad_data.get("has_defect"),
                "region": ad_data.get("region"),
                "area_ratio": ad_data.get("area_ratio"),
            },
        }
        ad_info_text = format_ad_info(ad_info)

        report_instruction: str | None = None
        if self.domain_rag_service is not None:
            try:
                report_instruction = self.domain_rag_service.build_report_instruction(
                    model_category=category,
                    ad_data=ad_data,
                    ad_info_text=ad_info_text,
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

        llm_report = result.get("llm_report")
        if isinstance(llm_report, dict):
            meta = dict(llm_report.get("_metadata", {})) if isinstance(llm_report.get("_metadata"), dict) else {}
            meta.update(
                {
                    "prompt_mode": "domain_rag" if report_instruction else "ad_only",
                    "ad_info_used": ad_info,
                    "reference_image_path": few_shot_paths[0] if few_shot_paths else None,
                }
            )
            llm_report = dict(llm_report)
            llm_report["_metadata"] = meta

        return {
            "is_anomaly_llm": result.get("is_anomaly_LLM"),
            "llm_report": llm_report,
            "llm_summary": result.get("llm_summary"),
            "llm_inference_duration": result.get("llm_inference_duration"),
            "ad_decision": ad_decision_raw.lower(),
        }
