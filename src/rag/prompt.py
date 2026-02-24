from __future__ import annotations
from typing import Dict, List, Optional
from langchain_core.documents import Document


INSTRUCTION_RAG = '''
You are an industrial inspector who checks products by images. You should judge whether there is a defect in the query image and answer the questions about it.
An anomaly detection model has analyzed this image and provided the following information:
{ad_info}

Here is the domain knowledge about known defect types for this product:
{domain_knowledge}

Use the anomaly detection results and domain knowledge along with your visual analysis to answer the questions.
Answer with the option's letter from the given choices directly!

Finally, you should output a list of answer, such as:
1. Answer: B.
2. Answer: B.
3. Answer: A.
...
'''

# -- 위 prompt 내용 --

# 너는 이미지로 제품을 검사하는 산업 검사관이야.                                                       
# 쿼리 이미지에 결함이 있는지 판단하고 질문에 답해.                                                  
                                                                                                    
# 이상 탐지 모델이 이 이미지를 분석한 결과야:                                                          
# {ad_info}                                                                                            
                                                                                                    
# 이 제품의 알려진 결함 유형에 대한 도메인 지식이야:
# {domain_knowledge}

# 이상 탐지 결과와 도메인 지식, 그리고 네 시각 분석을 활용해서 답해.
# 선택지의 알파벳으로 바로 답해!

# 답변은 이렇게 출력해:
# 1. Answer: B.
# 2. Answer: B.
# 3. Answer: A.



REPORT_PROMPT_RAG = '''당신은 제조 품질관리 수석 검사관입니다.
제품 카테고리: {category}

AD 사전 분석:
{ad_info}

도메인 지식:
{domain_knowledge}

판정 원칙:
1) 이미지 시각 근거를 최우선으로 사용하세요.
2) AD와 도메인 지식은 보조 근거로 사용하세요.
3) 근거가 약하면 "is_anomaly"를 false로 두고 confidence를 낮게 설정하세요.
4) 이미지에 없는 결함/원인/위치 추측을 금지합니다.

RAG 규칙:
- 도메인 지식은 결함명 구체화/원인 추정 보조에만 사용하세요.
- 이미지 근거 없이 도메인 지식을 복사하거나 단정하지 마세요.
- 도메인 지식과 시각 근거가 충돌하면 시각 근거를 우선하세요.
- AD 판정이 ANOMALY/NORMAL이고 decision_confidence>=0.90인 강한 신호에서 정반대 판정을 내릴 경우:
  confidence는 0.55 이하로 제한하고, description에 시각적 반증 근거와 "AD와 충돌"을 명시하세요.

출력 규칙:
- JSON만 출력하세요(설명 문장/코드블록 금지).
- 아래 스키마의 key 이름을 그대로 사용하세요.
- 문자열 value는 한국어로 작성하세요.
- "severity", "risk_level"은 low/medium/high/none 중 하나만 사용하세요.
- "confidence"는 0.00~1.00 범위 숫자여야 합니다.

일관성 규칙:
- "is_anomaly"=false이면 anomaly_type/severity/location/possible_cause/risk_level은 모두 "none".
- "is_anomaly"=true이면 anomaly_type은 "none" 금지, severity/risk_level은 "none" 금지.
- anomaly 설명에는 최소 2개의 구체적 시각 근거(무엇이, 어디에)를 포함하세요.

결함 taxonomy:
scratch, crack, dent, deformation, contamination, foreign_material, seal_defect,
label_defect, print_defect, missing_part, misalignment, color_stain, other, none

반드시 아래 JSON 형식으로만 출력하세요:
{{{{
  "is_anomaly": true or false,
  "report": {{{{
    "anomaly_type": "specific defect type or none",
    "severity": "low/medium/high/none",
    "location": "defect location or none",
    "description": "근거 중심의 상세 설명(한국어)",
    "possible_cause": "가장 가능성 높은 원인 또는 none",
    "confidence": 0.0 to 1.0,
    "recommendation": "구체적 시정/예방 조치(한국어)"
  }}}},
  "summary": {{{{
    "summary": "정확히 3문장: 최종 판정 + 핵심 근거/긴급도 + 구체적 시정/예방조치(한국어)",
    "risk_level": "low/medium/high/none"
  }}}}
}}}}'''


def report_prompt_rag(
    category: str,
    domain_knowledge: str = "",
    ad_info: str = "",
) -> str:
    """Build a RAG-augmented report generation prompt.

    Args:
        category: Product category string.
        domain_knowledge: Formatted domain knowledge from retriever.format_context().
        ad_info: Formatted AD model output string.

    Returns:
        Filled report prompt string.
    """
    return REPORT_PROMPT_RAG.format(
        category=category,
        ad_info=ad_info or "No anomaly detection information available.",
        domain_knowledge=domain_knowledge or "No relevant domain knowledge found.",
    )


def rag_prompt(
    ad_info: str = "",
    domain_knowledge: str = "",
) -> str:
    """Build a complete RAG-augmented instruction prompt.

    Args:
        ad_info: Formatted AD model output string.
        domain_knowledge: Formatted domain knowledge from retriever.format_context().

    Returns:
        Filled instruction prompt string.
    """
    return INSTRUCTION_RAG.format(
        ad_info=ad_info or "No anomaly detection information available.",
        domain_knowledge=domain_knowledge or "No relevant domain knowledge found.",
    )
