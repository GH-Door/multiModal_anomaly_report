# 우정님 인수인계 문서 (LLM MCQ 평가 담당)

## 1. 담당 범위
- 목표: AD 결과를 활용해 MCQ 정확도 측정/비교
- 핵심: AD 포맷이 바뀌어도 평가 코드 최소 수정으로 유지

## 2. 반드시 알아야 할 파일
- 실행 엔트리
  - `scripts/run_experiment.py`
- AD 입력 파서/어댑터
  - `src/ad/io.py`
  - `src/ad/adapter.py`
- LLM 클라이언트 팩토리
  - `src/mllm/factory.py`
- 프롬프트 반영
  - `src/mllm/base.py` (`format_ad_info`)
- 평가 지표
  - `src/eval/metrics.py`

## 3. 호출 체인 (실제 함수 기준)
1) `run_experiment(cfg)` in `scripts/run_experiment.py`
2) AD 결과 로드
- `load_ad_predictions_file(path)` in `src/ad/io.py`
- 내부에서 `parse_ad_predictions_payload()` + `normalize_image_key()` 처리
3) 이미지별 AD 변환
- `to_llm_ad_info(prediction)` in `src/ad/adapter.py`
- 결과 키: `is_anomaly`, `anomaly_score`, `defect_location` (+ optional)
4) LLM 호출
- `llm_client.generate_answers(..., ad_info=...)`
5) 저장/평가
- answers JSON 저장
- `calculate_accuracy_mmad()` 실행

## 4. AD JSON 계약 (평가에서 실제로 쓰는 것)
필수:
- `image_path`
- `anomaly_score`
- `is_anomaly` 또는 대체키(`model_is_anomaly`, `decision`)
- `defect_location` (`has_defect`, `bbox`, `region`, `area_ratio`)

선택:
- `confidence`, `reason_codes`, `report_guidance`, `use_location_for_report`

참고:
- policy 객체를 LLM 입력으로 직접 쓰지 않음
- 출력 메타의 `decision_rules_version`은 추적용

## 5. 실행 스크립트
### A) AD + LLM 자동 실행
```bash
python scripts/run_experiment.py --config configs/experiment.yaml
```

### B) AD 결과 재사용 (반복 실험 권장)
```bash
python scripts/run_experiment.py \
  --config configs/experiment.yaml \
  --ad-model patchcore \
  --ad-output outputs/eval/patchcore_predictions.json
```

### C) 빠른 smoke test
```bash
python scripts/run_experiment.py --config configs/experiment.yaml --max-images 10
```

## 6. 실험 제안 (우선순위)
1) 안정화 실험
- `--max-images 10`으로 파이프라인 정상 동작 확인
- 결과 파일: `outputs/eval/answers_*.json`, `*.meta.json`

2) 재현성 실험
- `sample_per_folder`, `sample_seed` 고정
- LLM 모델만 바꿔 비교

3) AD 영향도 실험
- 조건 A: `--ad-model null`
- 조건 B: `--ad-model patchcore --ad-output ...`
- 두 결과 정확도 차이 비교

## 7. 자주 깨지는 포인트
1) AD 매칭 누락
- 증상: `ad_info`가 비어 LLM에 전달
- 확인: `image_path` 키 포맷, slash 정규화

2) AD 모델 0개 인식
- 증상: `Available models: 0`, `Skipped (no model)`
- 확인: 체크포인트 경로/버전(`ad.version`)

3) InternVL 로딩 실패
- 증상: `all_tied_weights_keys`
- 조치: `transformers==4.52.4` 고정

## 8. 우정님 작업 시 수정 금지(권장)
- `eval_llm_baseline.py`는 타 담당 작업과 충돌 가능하므로 변경 금지
- AD 출력 스키마를 직접 파싱하지 말고 `src/ad/*`만 사용
