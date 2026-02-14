# 성호님 인수인계 문서 (서비스 파이프라인 담당)

## 1) 목표 파이프라인
서비스 목표는 아래 1줄입니다.

`입력 이미지(1장/배치) -> PatchCore AD 추론(384) -> LLM-friendly JSON 생성 -> 원본 이미지+JSON을 MLLM에 입력 -> 리포트 생성`

현재 저장소에서 이 흐름의 핵심은 이미 준비되어 있고, 서비스 오케스트레이션만 붙이면 됩니다.

## 2) 바로 재사용 가능한 코드
- AD 실행 엔트리: `scripts/run_ad_inference.py`
- AD 결과 -> LLM 입력 변환: `src/ad/adapter.py` (`to_llm_ad_info`)
- AD JSON 로드/정규화: `src/ad/io.py`
- LLM 호출: `src/mllm/factory.py` + 각 클라이언트 (`generate_report`, `generate_answers`)
- 실험 오케스트레이션 참고: `scripts/run_experiment.py`

## 3) AD 출력 JSON 계약 (중요)
`run_ad_inference.py --output-format report` 기준:
- top-level
  - `schema_version`: `ad_report_v1`
  - `bbox_spec.reference`: `original_image_pixels`
  - `predictions`: 이미지별 결과 배열
- per-image 주요 필드
  - `image_path`
  - `anomaly_score`
  - `decision` (`normal | anomaly | review_needed`)
  - `defect_location.bbox` (`[x_min, y_min, x_max, y_max]`, 원본 픽셀 기준)
  - `reason_codes`, `report_guidance`, `use_location_for_report`

주의: `policy` 객체는 출력 JSON에 직접 노출하지 않습니다.

## 4) 서비스 구현 시 권장 순서
1. API/Worker 시작 시 AD 러너와 LLM 클라이언트를 1회 초기화.
2. 요청 이미지 전처리(경로 정리, 배치 구성) 후 AD 수행.
3. AD raw 결과를 `to_llm_ad_info()`로 LLM 입력 형태로 변환.
4. 원본 이미지 + AD 정보로 `generate_report()` 호출.
5. 최종 응답 JSON에 AD 요약 + 리포트를 함께 반환.

## 5) 속도 이슈 대응 포인트
- I/O 병목 완화: `--io-workers` 조정, 로컬 디스크 우선
- 디코드 옵션: `--decode-reduced`(정확도-속도 트레이드오프)
- GPU 최적화: `--batch-size`, `--amp`, `--postprocess-map input`
- 장시간 실행 안정화: `--gc-interval`, `--profile-interval`로 모니터링

## 6) PatchCore 전용 학습 정책
- 학습 스크립트: `scripts/train_anomalib.py`
- 현재는 PatchCore만 허용하도록 고정됨 (`anomaly.model != patchcore`면 예외)

## 7) 팀 간 계약 포인트
- 우정님(MCQ)은 `run_experiment.py` 인터페이스를 사용하므로,
  서비스 쪽은 `ad_info` 핵심 필드(`is_anomaly`, `anomaly_score`, `defect_location`) 호환만 유지하면 됨.
- `eval_llm_baseline.py`는 타 담당 파일이므로 수정 없이 유지.

## 8) 코랩 검증 노트북
- `notebooks/lee/colab_train_eval_validation.ipynb`
- 스크립트 동작 검증(학습/추론/평가) 순서가 정리되어 있어 재현 테스트에 바로 사용 가능
