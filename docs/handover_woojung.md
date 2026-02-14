# 우정님 인수인계 문서 (MCQ 평가 담당)

## 1) 목적
이 문서는 `AD(JSON) -> LLM -> MCQ 평가` 흐름을 **코드 수정 최소화**로 안정적으로 돌리기 위한 운영 가이드입니다.
핵심 실행 파일은 `scripts/run_experiment.py` 입니다.

## 2) 지금 고정된 인터페이스 (깨지면 안 됨)
- AD 결과 파일 입력: `--ad-output <json>`
- AD 결과 파싱: `src/ad/io.py`
  - `load_ad_predictions_file()`
  - `normalize_image_key()`
- LLM 입력용 변환: `src/ad/adapter.py`
  - `to_llm_ad_info()`
- LLM 프롬프트에 실제 반영: `src/mllm/base.py`의 `format_ad_info()`

즉, AD JSON 구조가 일부 달라도 `src/ad/io.py`와 `src/ad/adapter.py`에서 흡수하도록 설계되어 있습니다.

## 3) 실행 순서
### A. AD + LLM 한번에
```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```
- 내부에서 `scripts/run_ad_inference.py`를 호출해 AD JSON을 먼저 생성
- 생성된 AD 결과를 LLM 평가에 자동 연결

### B. 기존 AD JSON 재사용 (권장)
```bash
python scripts/run_experiment.py \
  --config configs/experiment.yaml \
  --ad-model patchcore \
  --ad-output output/ad_predictions.json
```
- AD 재추론 없이 바로 MCQ 평가
- 반복 실험 속도 훨씬 빠름

## 4) AD JSON에서 LLM이 실제로 쓰는 필드
- `anomaly_score`
- `is_anomaly` (없으면 `decision`/`model_is_anomaly` 등에서 보정)
- `defect_location` (`has_defect`, `bbox`, `region`, `area_ratio`)
- 선택 필드: `reason_codes`, `report_guidance`, `confidence`

참고: `policy` 객체 자체는 출력 JSON에 직접 넣지 않습니다. 대신 `decision_rules_version`만 상단 메타에 남깁니다.

## 5) 결과물 경로
- 정답/응답: `outputs/eval/answers_*.json`
- 메타: `outputs/eval/answers_*.meta.json`
- 정확도 로그: `scripts/run_experiment.py` 실행 시 콘솔 출력 + 메타 파일

## 6) 자주 나는 이슈
- AD 매칭 누락: 경로 구분자 차이(`\\` vs `/`) -> `normalize_image_key()`가 처리
- AD 정보가 프롬프트에 안 들어감: `ad_output` 경로 오타/키 불일치 확인
- 느린 실험: `--ad-output` 재사용 + `--max-images`로 스모크 테스트 먼저

## 7) 코랩 검증 노트북
- `notebooks/lee/colab_train_eval_validation.ipynb`
- 학습/AD 추론/실험 실행을 순서대로 검증 가능
