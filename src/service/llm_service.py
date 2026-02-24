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
        ad_decision_raw = str(ad_data.get("ad_decision", "review_needed"))
        few_shot_paths: list[str] = []
        if ref_path:
            resolved = _resolve_few_shot_path(ref_path)
            if resolved:
                few_shot_paths = [resolved]
            else:
                logger.warning("Skipping unavailable RAG reference image for LLM: %s", ref_path)

        decision_confidence = ad_data.get("decision_confidence")
        if decision_confidence is None:
            # Backward compatibility: some payloads used a scalar confidence field.
            fallback_confidence = ad_data.get("confidence")
            if isinstance(fallback_confidence, (int, float, str)):
                decision_confidence = fallback_confidence

        ad_info = {
            "anomaly_score": ad_data.get("ad_score", 0.0),
            "decision": ad_decision_raw.lower(),
            "is_anomaly": ad_decision_raw.lower() == "anomaly",
            "decision_confidence": decision_confidence,
            "reason_codes": ad_data.get("reason_codes"),
            "defect_location": {
                "has_defect": ad_data.get("has_defect"),
                "region": ad_data.get("region"),
                "area_ratio": ad_data.get("area_ratio"),
            },
        }
        confidence_raw = ad_data.get("confidence")
        if isinstance(confidence_raw, dict):
            ad_info["confidence"] = confidence_raw
        else:
            try:
                if confidence_raw is not None:
                    ad_info["location_confidence"] = float(confidence_raw)
            except (TypeError, ValueError):
                pass

        policy_info: Dict[str, Any] = {}
        if isinstance(policy, dict):
            for key in (
                "reliability",
                "ad_weight",
                "review_band",
                "t_low",
                "t_high",
                "location_mode",
                "min_location_confidence",
                "use_bbox",
            ):
                value = policy.get(key)
                if value is not None:
                    policy_info[key] = value
        if policy_info:
            ad_info["policy"] = policy_info
            if "confidence" not in ad_info and isinstance(policy_info.get("reliability"), str):
                ad_info["confidence"] = {"reliability": str(policy_info["reliability"]).strip().lower()}

        if isinstance(ad_data.get("report_guidance"), dict):
            ad_info["report_guidance"] = ad_data["report_guidance"]
        if isinstance(ad_data.get("decision_basis"), dict):
            ad_info["decision_basis"] = ad_data["decision_basis"]
        if "review_needed" in ad_data:
            ad_info["review_needed"] = bool(ad_data.get("review_needed"))
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
