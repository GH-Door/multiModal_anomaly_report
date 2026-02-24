from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from src.mllm.base import format_ad_info

logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool_like(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in {"true", "1", "yes", "y", "anomaly", "defect", "bad", "ng"}:
        return True
    if s in {"false", "0", "no", "n", "normal", "good", "ok"}:
        return False
    return None


def _strong_ad_signal(ad_data: Dict[str, Any], policy: Dict[str, Any], ad_decision: str) -> bool:
    ad_decision_norm = str(ad_decision or "").strip().lower()
    if ad_decision_norm not in {"anomaly", "normal"}:
        return False

    basis = ad_data.get("decision_basis")
    if not isinstance(basis, dict):
        basis = {}

    score = _safe_float(ad_data.get("ad_score"))
    score_delta = abs(float(_safe_float(basis.get("score_delta")) or 0.0))
    decision_conf = float(_safe_float(ad_data.get("decision_confidence")) or 0.0)

    reliability = str(policy.get("reliability", "medium")).strip().lower()
    min_conf = {"high": 0.65, "medium": 0.72, "low": 0.82}.get(reliability, 0.72)
    min_delta = {"high": 0.05, "medium": 0.08, "low": 0.12}.get(reliability, 0.08)
    margin = {"high": 0.05, "medium": 0.08, "low": 0.12}.get(reliability, 0.08)

    if ad_decision_norm == "anomaly":
        t_high = _safe_float(policy.get("t_high"))
        if score is not None and t_high is not None and score >= float(t_high) + margin:
            return True
    else:
        t_low = _safe_float(policy.get("t_low"))
        if score is not None and t_low is not None and score <= float(t_low) - margin:
            return True

    return bool(decision_conf >= min_conf and score_delta >= min_delta)


def _is_strong_ad_llm_conflict(
    llm_result: Dict[str, Any],
    *,
    ad_data: Dict[str, Any],
    policy: Dict[str, Any],
    ad_decision: str,
) -> bool:
    ad_decision_norm = str(ad_decision or "").strip().lower()
    llm_flag = _to_bool_like(llm_result.get("is_anomaly_LLM"))
    if llm_flag is None:
        return False
    if ad_decision_norm == "anomaly" and llm_flag is False:
        return _strong_ad_signal(ad_data, policy, ad_decision_norm)
    if ad_decision_norm == "normal" and llm_flag is True:
        return _strong_ad_signal(ad_data, policy, ad_decision_norm)
    return False


def _has_llm_payload(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("is_anomaly_LLM") is not None:
        return True
    llm_report = result.get("llm_report")
    if isinstance(llm_report, dict) and any(
        key in llm_report
        for key in ("anomaly_type", "severity", "location", "description", "possible_cause", "recommendation")
    ):
        return True
    if isinstance(result.get("llm_summary"), dict) and result.get("llm_summary"):
        return True
    if isinstance(result.get("llm_summary"), str) and str(result.get("llm_summary")).strip():
        return True
    return False


def _apply_consistency_guard(
    result: Dict[str, Any],
    *,
    ad_data: Dict[str, Any],
    policy: Dict[str, Any],
    ad_decision: str,
) -> Dict[str, Any]:
    out = dict(result)
    llm_flag = _to_bool_like(out.get("is_anomaly_LLM"))
    if llm_flag is None:
        return out

    ad_decision_norm = str(ad_decision or "").strip().lower()
    strong = _strong_ad_signal(ad_data, policy, ad_decision_norm)
    if not strong:
        return out

    llm_report = out.get("llm_report")
    if isinstance(llm_report, dict):
        report = dict(llm_report)
        if ad_decision_norm == "anomaly" and llm_flag is False:
            conf = _safe_float(report.get("confidence"))
            if conf is None or conf > 0.55:
                report["confidence"] = 0.55
            recommendation = str(report.get("recommendation", "")).strip().lower()
            if recommendation in {"", "none", "없음", "해당 없음"}:
                report["recommendation"] = "수동 재검수(원본 확대 확인 및 동일 라인 샘플 재검사)를 진행하세요."
            desc = str(report.get("description", "")).strip()
            note = "AD 강한 이상 신호와 충돌하여 수동 검토가 필요합니다."
            if note not in desc:
                report["description"] = f"{desc} {note}".strip()
        out["llm_report"] = report
    return out


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
        used_retry = False
        retry_reason: str | None = None

        if report_instruction and _is_strong_ad_llm_conflict(
            result,
            ad_data=ad_data,
            policy=policy,
            ad_decision=ad_decision_raw,
        ):
            retry_reason = "strong_ad_llm_conflict_after_domain_rag"
            logger.info(
                "Retry LLM report without domain RAG due strong AD conflict | category=%s ad_decision=%s",
                category,
                str(ad_decision_raw).lower(),
            )
            retry_result = self.client.generate_report(
                image_path=image_path,
                category=category,
                ad_info=ad_info,
                few_shot_paths=few_shot_paths,
                instruction=None,
            )
            if _has_llm_payload(retry_result):
                result = retry_result
                used_retry = True

        result = _apply_consistency_guard(
            result,
            ad_data=ad_data,
            policy=policy,
            ad_decision=ad_decision_raw,
        )

        llm_report = result.get("llm_report")
        if isinstance(llm_report, dict):
            meta = dict(llm_report.get("_metadata", {})) if isinstance(llm_report.get("_metadata"), dict) else {}
            prompt_mode = "domain_rag" if report_instruction else "ad_only"
            if used_retry:
                prompt_mode = "domain_rag_retry_ad_only"
            meta.update(
                {
                    "prompt_mode": prompt_mode,
                    "ad_info_used": ad_info,
                    "reference_image_path": few_shot_paths[0] if few_shot_paths else None,
                    "retry_used": used_retry,
                    "retry_reason": retry_reason,
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
