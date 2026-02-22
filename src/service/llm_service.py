from __future__ import annotations

from typing import Any, Dict


class LlmService:
    """LLM report adapter returning DB-compatible keys."""

    def __init__(self, client) -> None:
        self.client = client

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
        ad_info = {
            "anomaly_score": ad_data.get("ad_score", 0.0),
            "is_anomaly": ad_decision_raw.lower() == "anomaly",
            "defect_location": {
                "region": ad_data.get("region"),
                "area_ratio": ad_data.get("area_ratio"),
            },
        }
        result = self.client.generate_report(
            image_path=image_path,
            category=category,
            ad_info=ad_info,
            few_shot_paths=[ref_path] if ref_path else [],
        )
        return {
            "is_anomaly_llm": result.get("is_anomaly_LLM"),
            "llm_report": result.get("llm_report"),
            "llm_summary": result.get("llm_summary"),
            "llm_inference_duration": result.get("llm_inference_duration"),
            "ad_decision": ad_decision_raw.lower(),
        }
