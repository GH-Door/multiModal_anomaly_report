# 성호님 인수인계 문서 (서비스 파이프라인 구현 담당)

## 1. 구현 목표
서비스 파이프라인:
`입력 이미지(1장/배치) -> AD(PatchCore) -> AD JSON -> 원본 이미지 + AD 정보를 MLLM 입력 -> 리포트 생성`

핵심 원칙:
- 모듈 API 재사용
- AD/LLM 변경이 서로 덜 깨지게 분리

## 2. 필수 파일 맵
- AD 실행
  - `scripts/run_ad_inference.py`
- AD 결과 I/O/표준화
  - `src/ad/io.py`
  - `src/ad/adapter.py`
- LLM 실행
  - `src/mllm/factory.py`
  - `src/mllm/base.py` (`generate_report`, `generate_answers`)
- 참조 오케스트레이션
  - `scripts/run_experiment.py`

## 3. 호출 함수와 역할
### AD 추론
- CLI: `scripts/run_ad_inference.py`
- 내부 핵심:
  - `PatchCoreCheckpointRunner.predict_batch(...)`
  - `build_result_dict(...)` (image별 결과 JSON 구성)
  - `save_results(...)`

### AD JSON 로드/변환
- `load_ad_predictions_file(path)` in `src/ad/io.py`
- `to_llm_ad_info(pred)` in `src/ad/adapter.py`

### LLM 리포트 생성
- `get_llm_client(model_name, ...)` in `src/mllm/factory.py`
- `client.generate_report(image_path, category, ad_info=...)` in `src/mllm/base.py`

## 4. AD JSON 스키마 (서비스에서 신뢰해야 할 키)
Top-level (`output-format=report`):
- `schema_version`, `backend`, `model_threshold`, `bbox_spec`, `predictions`

Per-image:
- `image_path`, `dataset`, `category`
- `anomaly_score`, `decision`, `review_needed`
- `defect_location` (bbox 포함)
- `report_guidance`, `reason_codes`, `use_location_for_report`

bbox 기준:
- `bbox_spec.reference = original_image_pixels`
- 형식: `xyxy` = `[x_min, y_min, x_max, y_max]`

## 5. 서비스 구현 제안 (단계별)
1) Phase 1: Inference service wrapper
- 입력: 이미지 path/list
- 출력: image별 AD result dict
- 구현: `run_ad_inference.py`의 핵심 로직을 모듈 호출로 감싸는 thin wrapper

2) Phase 2: AD -> LLM adapter layer
- `to_llm_ad_info()`로 최소 표준 필드만 전달
- LLM 프롬프트는 `format_ad_info()`를 통일 사용

3) Phase 3: Orchestrator
- image batch 순회
- AD 결과 캐시(옵션)
- LLM 호출 후 report JSON 합치기

4) Phase 4: Evaluation hook
- 동일 입력을 `run_experiment.py` 경로로도 재현 가능하게 유지
- 실험/서비스 결과 비교 가능하도록 output schema 통일

## 6. 실행 예시
### AD만 실행
```bash
python scripts/run_ad_inference.py \
  --backend ckpt \
  --checkpoint-dir /content/.../checkpoints/patchcore_384 \
  --data-root /content/.../MMAD \
  --mmad-json /content/.../mmad_10classes.json \
  --output outputs/eval/patchcore_predictions.json \
  --output-format report \
  --device cuda --batch-size 16 --amp
```

### End-to-end 평가 재현
```bash
python scripts/run_experiment.py --config configs/experiment.yaml --max-images 10
```

## 7. 성능/안정화 포인트
1) AD 속도
- `--batch-size`, `--io-workers`, `--amp`, `--postprocess-map input` 조정
- 드라이브 I/O 병목 시 로컬 디스크 staging 권장

2) 체크포인트 탐색 실패
- `ad.version` 고정 시 해당 버전 폴더 존재 확인
- 없으면 `ad.version: null`로 최신 사용

3) LLM 모델 로드 실패
- InternVL은 transformers 버전 민감
- 코랩: `transformers==4.52.4` 권장

## 8. 팀 인터페이스 계약
- 우정님(MCQ)은 `run_experiment.py` 기준으로 평가하므로,
  서비스 구현에서도 `to_llm_ad_info()` 출력 키를 유지해야 함.
- 타 팀 작업 파일(`eval_llm_baseline.py`)은 수정하지 않는 것을 기본 원칙으로 유지.

## 9. 구현 체크리스트
- [ ] 단일 이미지 입력/출력
- [ ] 배치 입력/출력
- [ ] AD 실패 시 graceful fallback(LLM-only 또는 에러 코드)
- [ ] bbox 좌표 기준 명시(원본 픽셀)
- [ ] 로그/메타 저장(재현성)
