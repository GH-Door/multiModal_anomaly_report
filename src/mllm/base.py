"""Base class for LLM clients following MMAD paper evaluation protocol."""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)

# MMAD paper's instruction prompt
INSTRUCTION = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# Instruction with AD model output
INSTRUCTION_WITH_AD = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.

An anomaly detection model has pre-analyzed this image. Here is the result:
{ad_info}

Consider this as a reference, but rely primarily on your own visual analysis. The model can make mistakes.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# ── Report generation prompts ──────────────────────────────────────────

REPORT_PROMPT = '''당신은 제조 품질관리 수석 검사관입니다.
목표는 단일 제품 이미지로 고품질 검사 리포트를 작성하는 것입니다.

제품 카테고리: {category}

판정 우선순위:
1) 이미지에서 확인되는 시각적 근거(최우선)
2) 일반 결함 지식
3) 기타 단서

품질 규칙:
- 눈으로 확인 가능한 구체적 결함 근거가 있을 때만 "is_anomaly"를 true로 설정하세요.
- 근거가 약하거나 모호하면 "is_anomaly"를 false로 두고 confidence를 낮게 설정하세요.
- 이미지에 보이지 않는 결함/원인/위치를 추측해 만들어내지 마세요.

출력 언어/형식 규칙:
- JSON만 출력하세요(마크다운, 코드블록, 설명 문장 금지).
- JSON key 이름은 아래 스키마와 정확히 동일하게 유지하세요.
- 모든 문자열 value는 한국어로 작성하세요.
- 단 "severity"와 "risk_level" 값은 low, medium, high, none 중 하나만 사용하세요.
- "confidence"는 0.00~1.00 범위의 숫자로 작성하세요.

필드 일관성 규칙:
- "is_anomaly"가 false인 경우:
  anomaly_type="none", severity="none", location="none", possible_cause="none", risk_level="none"으로 맞추세요.
  description에는 정상 판정의 시각적 근거를 작성하세요.
- "is_anomaly"가 true인 경우:
  anomaly_type은 구체적이어야 하며("none" 금지), severity/risk_level은 "none"이 아니어야 합니다.
  description에는 최소 2개의 구체적 시각 근거(무엇이, 어디에)를 포함하세요.

다음 결함 taxonomy를 참고해 가장 가까운 레이블을 선택하세요:
scratch, crack, dent, deformation, contamination, foreign_material, seal_defect,
label_defect, print_defect, missing_part, misalignment, color_stain, other, none

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "is_anomaly": true or false,
  "report": {{
    "anomaly_type": "specific defect type or none",
    "severity": "low/medium/high/none",
    "location": "defect location or none",
    "description": "근거 중심의 상세 설명(한국어)",
    "possible_cause": "가장 가능성 높은 원인 또는 none",
    "confidence": 0.0 to 1.0,
    "recommendation": "구체적 시정/예방 조치(한국어)"
  }},
    "summary": {{
    "summary": "정확히 3문장: 최종 판정 + 핵심 근거/긴급도 + 구체적 시정/예방조치(한국어)",
    "risk_level": "low/medium/high/none"
  }}
}}'''

REPORT_PROMPT_WITH_AD = '''당신은 제조 품질관리 수석 검사관입니다.
목표는 단일 제품 이미지로 고품질 검사 리포트를 작성하는 것입니다.

제품 카테고리: {category}

다음은 이상탐지(AD) 모델의 사전 분석 결과입니다:
{ad_info}

판정 우선순위:
1) 이미지에서 확인되는 시각적 근거(최우선)
2) AD 결과(보조 근거)
3) 일반 결함 지식

충돌 처리 규칙:
- AD 결과와 시각 근거가 충돌하면 시각 근거를 우선하세요.
- 중요한 충돌인 경우 description에 그 사실을 한국어로 간단히 명시하세요.

품질 규칙:
- 눈으로 확인 가능한 구체적 결함 근거가 있을 때만 "is_anomaly"를 true로 설정하세요.
- 근거가 약하거나 모호하면 "is_anomaly"를 false로 두고 confidence를 낮게 설정하세요.
- 이미지에 보이지 않는 결함/원인/위치를 추측해 만들어내지 마세요.

출력 언어/형식 규칙:
- JSON만 출력하세요(마크다운, 코드블록, 설명 문장 금지).
- JSON key 이름은 아래 스키마와 정확히 동일하게 유지하세요.
- 모든 문자열 value는 한국어로 작성하세요.
- 단 "severity"와 "risk_level" 값은 low, medium, high, none 중 하나만 사용하세요.
- "confidence"는 0.00~1.00 범위의 숫자로 작성하세요.

필드 일관성 규칙:
- "is_anomaly"가 false인 경우:
  anomaly_type="none", severity="none", location="none", possible_cause="none", risk_level="none"으로 맞추세요.
  description에는 정상 판정의 시각적 근거를 작성하세요.
- "is_anomaly"가 true인 경우:
  anomaly_type은 구체적이어야 하며("none" 금지), severity/risk_level은 "none"이 아니어야 합니다.
  description에는 최소 2개의 구체적 시각 근거(무엇이, 어디에)를 포함하세요.

다음 결함 taxonomy를 참고해 가장 가까운 레이블을 선택하세요:
scratch, crack, dent, deformation, contamination, foreign_material, seal_defect,
label_defect, print_defect, missing_part, misalignment, color_stain, other, none

반드시 아래 JSON 형식으로만 출력하세요:
{{
  "is_anomaly": true or false,
  "report": {{
    "anomaly_type": "specific defect type or none",
    "severity": "low/medium/high/none",
    "location": "defect location or none",
    "description": "근거 중심의 상세 설명(한국어)",
    "possible_cause": "가장 가능성 높은 원인 또는 none",
    "confidence": 0.0 to 1.0,
    "recommendation": "구체적 시정/예방 조치(한국어)"
  }},
  "summary": {{
    "summary": "정확히 3문장: 최종 판정 + 핵심 근거/긴급도 + 구체적 시정/예방조치(한국어)",
    "risk_level": "low/medium/high/none"
  }}
}}'''


def _parse_llm_json(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM response text.

    Handles common LLM quirks: escaped underscores, markdown fences, etc.
    """
    # Fix LLaVA-style escaped underscores
    cleaned = text.replace("\\_", "_")

    # Try to extract JSON object
    json_match = re.search(r'\{[\s\S]*\}', cleaned)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return None


def _normalize_decision(value: Any) -> bool:
    """Normalize various LLM is_anomaly outputs to bool."""
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    return s in ("true", "1", "yes", "anomaly", "이상", "불량", "defect", "bad", "abnormal")


def format_ad_info(ad_info: dict) -> str:
    """Format AD model output as a concise natural language summary.

    Args:
        ad_info: Dictionary containing AD model predictions.

    Returns:
        Formatted string describing the AD model's findings.
    """
    if not ad_info:
        return "No anomaly detection information available."

    lines = []

    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    score = _to_float(ad_info.get("anomaly_score"))
    decision = str(ad_info.get("decision", "")).strip().lower()
    is_anomaly = ad_info.get("is_anomaly")

    if decision in {"anomaly", "normal", "review_needed"}:
        decision_label = decision.upper()
        if score is not None:
            lines.append(f"- AD decision: {decision_label} (score: {score:.2f})")
        else:
            lines.append(f"- AD decision: {decision_label}")
    elif score is not None:
        status = "ANOMALOUS" if is_anomaly else "NORMAL"
        lines.append(f"- AD result: {status} (score: {score:.2f})")

    decision_conf = _to_float(ad_info.get("decision_confidence"))
    if decision_conf is not None:
        lines.append(f"- Decision confidence: {decision_conf:.2f} (0-1)")

    confidence = ad_info.get("confidence", {})
    if isinstance(confidence, dict):
        reliability = str(confidence.get("reliability", "")).strip().lower()
        if reliability:
            lines.append(f"- AD reliability for this class: {reliability}")

    guidance = ad_info.get("report_guidance", {})
    if isinstance(guidance, dict):
        use_for_decision = guidance.get("use_ad_for_anomaly_judgement")
        use_for_location = guidance.get("use_ad_for_location")
        if use_for_decision:
            lines.append(f"- Use AD for anomaly judgement: {use_for_decision}")
        if use_for_location:
            lines.append(f"- Use AD for location hints: {use_for_location}")

    reason_codes = ad_info.get("reason_codes")
    if isinstance(reason_codes, list) and reason_codes:
        joined = ", ".join(str(x) for x in reason_codes[:6])
        lines.append(f"- Caution flags: {joined}")

    loc = ad_info.get("defect_location", {})
    if isinstance(loc, dict) and loc.get("has_defect"):
        region = loc.get("region", "unknown")
        area = _to_float(loc.get("area_ratio"))
        lines.append(f"- Defect location hint: {region}")
        if area is not None and area > 0:
            lines.append(f"- Defect area hint: {area * 100:.1f}% of image")
    elif isinstance(loc, dict) and loc:
        lines.append("- No reliable localized defect hint")

    if not lines:
        return "No anomaly detection information available."

    return "\n".join(lines)


def get_mime_type(image_path: str) -> str:
    """Get MIME type from image path."""
    path_lower = image_path.lower()
    if path_lower.endswith(".png"):
        return "image/png"
    elif path_lower.endswith(".jpeg") or path_lower.endswith(".jpg"):
        return "image/jpeg"
    return "image/jpeg"


class BaseLLMClient(ABC):
    """Base class for MMAD LLM evaluation.

    Follows the exact protocol from the paper:
    - Few-shot normal templates + query image + questions
    - Answer parsing with regex + fuzzy matching fallback
    """

    def __init__(
        self,
        max_image_size: Tuple[int, int] = (512, 512),
        max_retries: int = 5,
        visualization: bool = False,
    ):
        self.max_image_size = max_image_size
        self.max_retries = max_retries
        self.visualization = visualization
        self.api_time_cost = 0.0

    def encode_image_to_base64(self, image) -> str:
        """Encode image to base64, resizing if necessary.

        Args:
            image: BGR image (numpy array) or path string
        """
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {image}")

        height, width = image.shape[:2]
        scale = min(
            self.max_image_size[0] / width,
            self.max_image_size[1] / height
        )

        if scale < 1.0:
            new_width, new_height = int(width * scale), int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        _, encoded = cv2.imencode('.jpg', image)
        return base64.b64encode(encoded).decode('utf-8')

    def parse_conversation(self, meta: dict) -> Tuple[List[Dict[str, str]], List[str], List[str]]:
        """Parse MMAD conversation format into questions, answers, and types.

        Returns:
            questions: List of {"type": "text", "text": "Question: ... \nA. ...\nB. ..."}
            answers: List of correct answer letters
            question_types: List of question type strings
        """
        questions = []
        answers = []
        question_types = []

        # Find conversation key
        for key in meta.keys():
            if key.startswith("conversation"):
                conversation = meta[key]
                for qa in conversation:
                    # Build options text
                    options = qa.get("Options", qa.get("options", {}))
                    options_text = ""
                    if isinstance(options, dict):
                        for opt_key in sorted(options.keys()):
                            options_text += f"{opt_key}. {options[opt_key]}\n"

                    question_text = qa.get("Question", qa.get("question", ""))
                    questions.append({
                        "type": "text",
                        "text": f"Question: {question_text} \n{options_text}"
                    })
                    answers.append(qa.get("Answer", qa.get("answer", "")))
                    question_types.append(qa.get("type", "unknown"))
                break

        return questions, answers, question_types

    def parse_answer(self, response_text: str, options: Optional[Dict[str, str]] = None) -> List[str]:
        """Parse answer letters from LLM response.

        Uses regex pattern matching with fuzzy matching fallback.
        """
        pattern = re.compile(r'\b([A-E])\b')
        found_answers = pattern.findall(response_text)

        if len(found_answers) == 0 and options is not None:
            pass  # Fallback to fuzzy matching
            options_values = list(options.values())
            closest_matches = get_close_matches(response_text, options_values, n=1, cutoff=0.0)
            if closest_matches:
                closest_match = closest_matches[0]
                for key, value in options.items():
                    if value == closest_match:
                        found_answers.append(key)
                        break

        return found_answers

    @abstractmethod
    def send_request(self, payload: dict) -> Optional[dict]:
        """Send request to LLM API. Must be implemented by subclass."""
        pass

    @abstractmethod
    def build_payload(
        self,
        query_image_path: str,
        few_shot_paths: List[str],
        questions: List[Dict[str, str]],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
        report_mode: bool = False,
    ) -> dict:
        """Build API payload. Must be implemented by subclass.

        Args:
            query_image_path: Path to the query image
            few_shot_paths: List of few-shot template image paths
            questions: List of question dictionaries
            ad_info: Optional anomaly detection model output dictionary
            instruction: Optional custom instruction (e.g. RAG prompt) to override default
            report_mode: If True, build payload for report generation (no MCQ phrasing)
        """
        pass

    def generate_answers(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers for all questions in the conversation.

        Following paper's protocol: ask questions incrementally.

        Args:
            query_image_path: Path to the query image
            meta: MMAD metadata dictionary
            few_shot_paths: List of few-shot template image paths
            ad_info: Optional anomaly detection model output dictionary
            instruction: Optional custom instruction (e.g. RAG prompt)

        Returns:
            questions: Parsed questions
            correct_answers: Ground truth answers
            predicted_answers: Model predictions (None if failed)
            question_types: Question type strings
        """
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        predicted_answers = []

        # Paper's approach: ask incrementally (1 question, then 2, then 3...)
        for i in range(len(questions)):
            part_questions = questions[:i + 1]
            payload = self.build_payload(query_image_path, few_shot_paths, part_questions, ad_info=ad_info, instruction=instruction)

            response = self.send_request(payload)
            if response is None:
                predicted_answers.append('')
                continue

            response_text = self.extract_response_text(response)
            parsed = self.parse_answer(response_text)

            if parsed:
                predicted_answers.append(parsed[-1])
            else:
                predicted_answers.append('')

        return questions, answers, predicted_answers, question_types

    def generate_answers_batch(
        self,
        query_image_path: str,
        meta: dict,
        few_shot_paths: List[str],
        ad_info: Optional[Dict] = None,
        instruction: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str], Optional[List[str]], List[str]]:
        """Generate answers for all questions in a single API call.

        More efficient than incremental, but may be less accurate.

        Args:
            query_image_path: Path to the query image
            meta: MMAD metadata dictionary
            few_shot_paths: List of few-shot template image paths
            ad_info: Optional anomaly detection model output dictionary
            instruction: Optional custom instruction (e.g. RAG prompt)

        Returns:
            questions: Parsed questions
            correct_answers: Ground truth answers
            predicted_answers: Model predictions (None if failed)
            question_types: Question type strings
        """
        questions, answers, question_types = self.parse_conversation(meta)

        if not questions or not answers:
            return questions, answers, None, question_types

        payload = self.build_payload(query_image_path, few_shot_paths, questions, ad_info=ad_info, instruction=instruction)
        response = self.send_request(payload)

        if response is None:
            return questions, answers, None, question_types

        response_text = self.extract_response_text(response)
        parsed = self.parse_answer(response_text)

        # Pad with empty strings if not enough answers
        while len(parsed) < len(questions):
            parsed.append('')

        return questions, answers, parsed[:len(questions)], question_types

    @abstractmethod
    def extract_response_text(self, response: dict) -> str:
        """Extract text content from API response. Must be implemented by subclass."""
        pass

    # ── Report generation ──────────────────────────────────────────────

    def build_report_payload(
        self,
        image_path: str,
        category: str,
        ad_info: Optional[Dict] = None,
        few_shot_paths: Optional[List[str]] = None,
        instruction: Optional[str] = None,
    ) -> dict:
        """Build payload for report generation.

        Subclasses may override this for model-specific formatting.
        Default implementation uses build_payload with an empty questions list
        and the report prompt as a single text question.
        """
        if instruction:
            prompt_text = instruction
        elif ad_info:
            prompt_text = REPORT_PROMPT_WITH_AD.format(
                category=category,
                ad_info=format_ad_info(ad_info),
            )
        else:
            prompt_text = REPORT_PROMPT.format(category=category)

        return self.build_payload(
            image_path,
            few_shot_paths or [],
            [],
            ad_info=ad_info,
            instruction=prompt_text,
            report_mode=True,
        )

    def generate_report(
        self,
        image_path: str,
        category: str,
        ad_info: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a structured inspection report for a single image.

        Args:
            image_path: Path to the product image.
            category: Product category string (e.g. "cigarette_box").
            ad_info: Optional dict with AD model results (score, is_anomaly, etc.).

        Returns:
            Dict with keys: is_anomaly_LLM, llm_report, llm_summary.
        """
        payload = self.build_report_payload(
            image_path,
            category,
            ad_info,
            few_shot_paths=kwargs.get("few_shot_paths"),
            instruction=kwargs.get("instruction"),
        )

        t0 = time.time()
        response = self.send_request(payload)
        inference_time = time.time() - t0

        # Default fallback
        result: Dict[str, Any] = {
            "is_anomaly_LLM": None,
            "llm_report": None,
            "llm_summary": None,
            "llm_inference_duration": round(inference_time, 3),
        }

        if response is None:
            logger.warning("LLM returned no response for %s", image_path)
            return result

        text = self.extract_response_text(response)
        parsed = _parse_llm_json(text)

        if parsed is None:
            logger.warning("Failed to parse JSON from LLM response: %s", text[:200])
            result["llm_report"] = {"raw_response": text}
            return result

        result["is_anomaly_LLM"] = _normalize_decision(parsed.get("is_anomaly", False))
        result["llm_report"] = parsed.get("report", parsed)
        result["llm_summary"] = parsed.get("summary")

        return result
