# 멀티모달 이상탐지 레포 전체 코드 해설서 (쉬운 버전)

이 문서는 `multimodal-anomaly-report-generation` 레포를 처음부터 끝까지 읽을 때, **어느 파일의 어느 줄에서 무슨 일이 일어나는지**를 빠르게 이해하도록 만든 가이드입니다.

핵심 목표는 3가지입니다.

1. 전체 파이프라인(학습/추론/평가/서비스)의 연결 구조를 이해한다.
2. 함수 단위로 입력/출력(실제 값 기준)을 이해한다.
3. 코드를 읽을 때 “어떤 순서로 어떤 줄을 보면 되는지”를 알 수 있게 한다.

---

## 1. 먼저 큰 그림: 이 레포는 무엇을 하는가?

이 레포는 크게 4단계입니다.

1. AD(Anomaly Detection) 모델 학습 또는 로드
2. 이미지에 대해 AD 추론 수행 -> AD JSON 생성
3. LLM이 이미지(+AD 정보)를 보고 MMAD 문항 답변/리포트 생성
4. 평가 지표 계산 또는 서비스(API/DB/UI)로 저장/조회

핵심 실행 스크립트는 아래 3개입니다.

1. `scripts/train_anomalib.py`: AD 학습/예측
2. `scripts/run_ad_inference.py`: AD 추론 결과(JSON) 생성
3. `scripts/run_experiment.py`: 샘플링 + (선택)AD + LLM + 평가 전체 오케스트레이션

---

## 2. 코드 읽는 순서 (가장 빠른 이해 경로)

전체 64개 파일을 한 번에 보면 부담이 큽니다. 아래 순서로 보면 가장 빠릅니다.

1. `scripts/run_experiment.py`
2. `scripts/run_ad_inference.py`
3. `src/ad/io.py`, `src/ad/adapter.py`
4. `src/mllm/base.py`, `src/mllm/factory.py`, `src/mllm/internvl_client.py`
5. `src/eval/metrics.py`
6. `src/service/pipeline.py`, `src/storage/pg.py`, `apps/api/main.py`
7. `scripts/train_anomalib.py`, `src/datasets/dataloader.py`
8. 나머지 파일은 부록 인덱스를 따라 참조

---

## 3. 파이프라인 데이터 구조 (실제 입출력 이해용)

### 3.1 MMAD 샘플 1건 (입력 데이터)

보통 아래 필드들이 핵심입니다.

- `image_path`: 예) `GoodsAD/cigarette_box/test/good/003_001.jpg`
- `dataset`: 예) `GoodsAD`
- `category`: 예) `cigarette_box`
- `conversation`: MCQ 질문/정답

### 3.2 AD 예측 1건 (`run_ad_inference.py` 출력)

대표 필드:

- `anomaly_score`: float
- `model_is_anomaly`: bool
- `decision`: `normal | anomaly | review_needed`
- `defect_location.bbox`: `[x1, y1, x2, y2]` 또는 `null`
- `report_guidance`: LLM 프롬프트에 넘길 정책 가이드
- `reason_codes`: 정책 근거 코드

### 3.3 LLM 답변 결과 1건 (`run_experiment.py` 출력)

대표 필드:

- 질문 리스트
- 정답 리스트
- 모델 예측 리스트(문자 선택지)
- 문항 타입별 성능 집계용 정보

---

## 4. 핵심 파일 라인별 해설 (상세)

아래는 실제로 “파일을 열었을 때 어디 줄을 먼저 읽어야 하는지” 기준입니다.

---

## 4.1 `scripts/run_experiment.py` (555 lines)

역할: 실험 전체 오케스트레이터

핵심 함수:

1. `stratified_sample(...)` L50-L79
2. `run_ad_inference(...)` L93-L174
3. `run_experiment(...)` L198-L441
4. `main()` L444-L551

라인 구간 독해 가이드:

1. L1-L47: import, 로깅, 경로 기본 설정
2. L50-L79 `stratified_sample`
- 입력: 전체 이미지 경로 목록, 폴더당 샘플 수
- 출력: 샘플된 이미지 경로 목록
- 핵심: 폴더(dataset/category/split) 단위 균등 샘플링
3. L82-L90: MMAD/AD JSON 로더 래퍼
4. L93-L174 `run_ad_inference`
- 내부에서 `scripts/run_ad_inference.py`를 서브프로세스로 호출
- 즉, 이 함수는 “AD 추론 자체”가 아니라 “AD 추론 스크립트를 실행”하는 부분
5. L177-L195 `resolve_paths`
- `config`, env, default 순으로 경로를 확정
6. L198-L441 `run_experiment`
- 핵심 실행 흐름
- 샘플링 -> 필요 시 AD 실행 -> LLM 클라이언트 생성 -> 질문 답변 생성 -> metric 계산 -> 결과 저장
7. L444-L551 `main`
- CLI 인자 처리
- yaml 설정 로드 + CLI override
- 최종 `run_experiment` 호출

실무 포인트:

1. `--ad-model null`이면 AD 단계 스킵
2. `--ad-model patchcore`이면 AD JSON 생성 후 LLM 입력에 연결
3. 속도 병목은 보통 LLM 구간

---

## 4.2 `scripts/run_ad_inference.py` (1421 lines)

역할: PatchCore 추론 + 정책 기반 후처리 + 리포트형 JSON 생성

핵심 함수/클래스:

1. 전처리/정책
- `compute_confidence_level` L165-L184
- `compute_defect_location` L187-L221
- `load_ad_policy` L265-L297
- `resolve_class_policy` L300-L317
- `decide_with_policy` L320-L328
- `build_reason_codes` L377-L409
- `build_llm_guidance` L412-L465
2. 추론 러너
- `PatchCoreCheckpointRunner` L468-L753
- `PatchCoreOnnxRunner` L756-L797
3. 결과 포맷
- `build_result_dict` L879-L994
- `build_output_payload` L832-L855
- `save_results` L858-L876
4. 엔트리포인트
- `main` L1020-L1417

라인 구간 독해 가이드:

1. L1-L104: CLI/환경 준비, 상수
2. L107-L162: 경로 파싱, 이미지 로딩 유틸
3. L165-L465: 정책 기반 판단 로직(신뢰도/리즌코드/LLM 가이드)
4. L468-L797: 실제 모델 실행 계층
- Checkpoint 러너: anomalib 체크포인트 로딩 + batch 예측
- ONNX 러너: export된 모델/메모리뱅크로 예측
5. L800-L876: 입력 이미지 수집 + 결과 payload 구성/저장
6. L879-L994: 단일 예측 결과를 표준 dict로 변환 (가장 중요한 조립 함수)
7. L997-L1017: config 해석
8. L1020-L1417: 카테고리별 루프, 추론, 예외 처리, 집계 출력

`build_result_dict(...)` 입력/출력 (핵심):

입력:

1. `image_path`: 상대 경로 문자열
2. `dataset`, `category`: 클래스 식별
3. `pred`: 모델 raw 결과 (`anomaly_score`, `anomaly_map`, `is_anomaly` 등)
4. `policy`: `configs/ad_policy.json`에서 읽은 정책 dict

출력:

1. `decision`, `review_needed`
2. `defect_location_raw`, `defect_location`
3. `report_guidance` (LLM이 AD를 얼마나 신뢰할지)
4. `reason_codes`

실무 포인트:

1. `Available models: 0`이면 대체로 checkpoint version 불일치
2. 느릴 때는 `read ms/img`와 `infer ms/img`를 분리해서 병목 판단
3. 서비스 재사용 시 가장 먼저 재사용할 함수는 `build_result_dict`, `build_output_payload`

---

## 4.3 `src/ad/io.py` (48 lines)

역할: AD 결과 JSON을 안정적으로 읽고, 이미지 경로 키로 인덱싱

핵심 함수:

1. `normalize_image_key` L10-L12
- 입력: `./GoodsAD\\drink_bottle\\...` 같은 경로
- 출력: `GoodsAD/drink_bottle/...` 표준 키
2. `parse_ad_predictions_payload` L27-L41
- 입력: report/list 형식 AD payload
- 출력: `{normalized_path: prediction_dict}`
3. `load_ad_predictions_file` L44-L47
- 입력: 파일 경로
- 출력: 위 인덱스 dict

실무 포인트:

1. Windows/Unix 경로 차이로 생기는 매칭 실패를 여기서 해결
2. `run_experiment.py`와 서비스 파이프라인 모두 재사용 가능

---

## 4.4 `src/ad/adapter.py` (82 lines)

역할: AD 결과를 LLM 프롬프트용 간결 스키마로 변환

핵심 함수:

1. `to_llm_ad_info` L25-L81

입력:

1. AD prediction dict (`anomaly_score`, `model_is_anomaly`, `defect_location`, `reason_codes` ...)

출력:

1. LLM 프롬프트에 넣기 좋은 축약 정보
- `is_anomaly`
- `anomaly_score`
- `defect_location`
- `reason_codes`
- `guidance`

실무 포인트:

1. AD 스키마 변경이 생겨도 이 파일만 수정하면 상위 코드 영향 최소화 가능

---

## 4.5 `src/mllm/base.py` (481 lines)

역할: 모든 LLM 클라이언트가 따르는 공통 프로토콜

핵심 요소:

1. `_parse_llm_json` L109-L124
2. `_normalize_decision` L127-L132
3. `format_ad_info` L135-L170
4. `BaseLLMClient` 클래스 L183-L481

`BaseLLMClient` 주요 메서드:

1. `parse_conversation` L226-L259
- 입력: MMAD conversation
- 출력: `questions`, `answers`, `q_types`
2. `generate_answers` L307-L357
- 1문항씩 호출
3. `generate_answers_batch` L359-L402
- 배치 호출(모델별로 override 가능)
4. `generate_report` L434-L481
- 입력: 이미지 경로, 카테고리, `ad_info`
- 출력: 구조화 리포트 dict

실무 포인트:

1. 새로운 모델 추가 시 `send_request`, `build_payload`, `extract_response_text` 구현이 핵심
2. InternVL/Qwen/LLaVA는 배치 모드 최적화가 이미 들어가 있음

---

## 4.6 `src/mllm/internvl_client.py` (511 lines)

역할: InternVL 로컬 추론 클라이언트

핵심 함수:

1. 이미지 전처리
- `build_transform` L49-L56
- `dynamic_preprocess` L76-L114
- `load_image` L117-L124
2. 모델 로딩
- `_load_model` L211-L261
3. 추론
- `generate_answers` L348-L430
- `generate_answers_batch` L432-L511

실무 포인트:

1. 느릴 때 가장 먼저 `batch_mode` 사용 여부 확인
2. 초기 가중치 로딩 시간이 매우 크므로 워밍업/캐시 전략 필요
3. transformers 버전 이슈 시 `all_tied_weights_keys` 관련 에러가 발생할 수 있음

---

## 4.7 `src/eval/metrics.py` (424 lines)

역할: MMAD 성능 지표 계산

핵심 함수:

1. `compute_anomaly_metrics` L133-L207
- AD 분류 성능(AUROC/F1/Recall/Precision 등)
2. `calculate_accuracy_mmad` L222-L401
- LLM 정답률 + Recall/Precision/F1 계산

LLM 평가의 Recall/Precision/F1 의미:

1. 기준 라벨: MMAD GT 문항에서 도출된 이상/정상 정답
2. 예측 라벨: 모델 답변에서 정규화된 이상/정상 판단
3. Recall: 실제 이상을 얼마나 놓치지 않았는지
4. Precision: 이상이라고 한 것 중 진짜 이상 비율
5. F1: Recall/Precision 균형 점수

---

## 4.8 `src/datasets/dataloader.py` (417 lines)

역할: GoodsAD/MVTec-LOCO 데이터셋/데이터모듈 구성

핵심 클래스:

1. `MVTecLOCODataset` L58-L163
2. `GoodsADDataset` L166-L263
3. `GoodsADDataModule` L266-L303
4. `MVTecLOCODataModule` L306-L353
5. `MMADLoader` L356-L417

실무 포인트:

1. 데이터 폴더 구조가 살짝만 달라도 `make_dataset`에서 바로 깨짐
2. 마스크 경로 규칙은 각 dataset 클래스에서 확인

---

## 4.9 `scripts/train_anomalib.py` (856 lines)

역할: PatchCore 학습/예측/체크포인트 버전 관리

핵심 클래스:

1. `Anomalibs` L86-L829
- `fit`, `fit_all`
- `predict`, `predict_all`
- `get_ckpt_path`, `get_version_dir` 등 버전 관리

실무 포인트:

1. 현재 구조는 PatchCore 중심으로 단순화되어 있음
2. 버전 디렉토리 규칙이 추론 스크립트와 맞아야 `Available models`가 정상

---

## 5. 서비스 관점에서 꼭 알아야 할 함수 (재사용 우선순위)

E2E 서비스(이미지 업로드 -> 결과 반환)를 만들 때 아래 순서로 재사용하면 됩니다.

1. AD 단계
- `scripts/run_ad_inference.py::build_result_dict`
- `scripts/run_ad_inference.py::build_output_payload`
2. AD 로드/매칭
- `src/ad/io.py::load_ad_predictions_file`
- `src/ad/io.py::normalize_image_key`
3. AD -> LLM 브리지
- `src/ad/adapter.py::to_llm_ad_info`
4. LLM 호출
- `src/mllm/factory.py::get_llm_client`
- `src/mllm/base.py::BaseLLMClient.generate_report`
5. 저장/응답
- `src/storage/pg.py::insert_report`
- `apps/api/main.py` 엔드포인트

---

## 6. 파일별 빠른 설명 (64개 전체)

아래는 전 파일을 빠르게 훑는 용도입니다.

### 6.1 scripts

1. `scripts/run_experiment.py`: 실험 오케스트레이터(샘플링, AD, LLM, metric)
2. `scripts/run_ad_inference.py`: AD 추론 및 표준 JSON 생성
3. `scripts/train_anomalib.py`: AD 학습/예측 파이프라인
4. `scripts/eval_llm_baseline.py`: LLM baseline 평가 스크립트
5. `scripts/eval_onnx.py`: ONNX AD 모델 성능 평가
6. `scripts/legacy/eval_onnx.py`: 구버전 ONNX 평가 스크립트
7. `scripts/export_patchcore.py`: PatchCore checkpoint -> ONNX/export
8. `scripts/visualize_bbox_mask_subset.py`: bbox/mask 시각화 및 품질 점검
9. `scripts/test_report_pipeline.py`: 단일 이미지 E2E 점검
10. `scripts/upload_reports_to_pg.py`: JSON 결과를 PostgreSQL에 업로드

### 6.2 src/ad

1. `src/ad/io.py`: AD 결과 파일 로드/정규화
2. `src/ad/adapter.py`: AD 결과를 LLM 입력 스키마로 변환

### 6.3 src/anomaly

1. `src/anomaly/base.py`: AD 모델 추상 베이스
2. `src/anomaly/patchcore_onnx.py`: ONNX PatchCore 추론 구현

### 6.4 src/mllm

1. `src/mllm/base.py`: 공통 베이스/파서/리포트 생성
2. `src/mllm/factory.py`: 모델명 -> 클라이언트 팩토리
3. `src/mllm/internvl_client.py`: InternVL 로컬 클라이언트
4. `src/mllm/qwen_client.py`: Qwen2.5-VL 클라이언트
5. `src/mllm/llava_client.py`: LLaVA 클라이언트
6. `src/mllm/openai_client.py`: OpenAI API 클라이언트
7. `src/mllm/gemini_client.py`: Gemini API 클라이언트
8. `src/mllm/claude_client.py`: Claude API 클라이언트
9. `src/mllm/echo.py`: 더미/샘플용 경량 구현
10. `src/mllm/__init__.py`: 클라이언트 생성 헬퍼 export

### 6.5 src/eval

1. `src/eval/metrics.py`: MMAD/AD 평가 지표 계산

### 6.6 src/datasets

1. `src/datasets/dataloader.py`: dataset/datamodule 생성
2. `src/datasets/mmad.py`: MMAD 샘플/템플릿 로더
3. `src/datasets/mmad_index_csv.py`: MMAD index CSV 파서/분할
4. `src/datasets/anomalib_folder_builder.py`: anomalib folder 데이터 생성
5. `src/datasets/preprocess.py`: 전처리 자리 파일

### 6.7 src/service/storage/apps

1. `src/service/pipeline.py`: E2E 서비스 파이프라인 클래스
2. `src/service/settings.py`: runtime 설정 로더
3. `src/storage/pg.py`: PostgreSQL CRUD
4. `src/storage/db.py`: DB 래퍼(구/경량)
5. `apps/api/main.py`: FastAPI 엔드포인트
6. `apps/dashboard/app.py`: 간단 대시보드 진입점
7. `apps/dashboard/app_gradio.py`: Gradio UI

### 6.8 src/rag/report/structure/visual/utils/common

1. `src/rag/indexer.py`: 지식문서 -> 벡터 인덱스
2. `src/rag/retriever.py`: 유사 문서 검색
3. `src/rag/prompt.py`: RAG 프롬프트 생성
4. `src/rag/embeddings.py`: 임베딩 모델 로더
5. `src/report/schema.py`: 리포트 스키마 로딩/검증
6. `src/structure/defect.py`: heatmap 기반 결함 구조화
7. `src/structure/render.py`: heatmap/overlay 저장
8. `src/visual/comparison.py`: GT vs 예측 시각화
9. `src/visual/plot.py`: 그래프/이미지 시각화 유틸
10. `src/utils/*`: 공통 유틸(로그, 경로, 설정, 장치, checkpoint, wandb)
11. `src/common/io.py`, `src/common/types.py`: 공통 I/O/타입

---

## 7. “한 줄 한 줄”로 읽는 실제 방법 (실전)

코드를 진짜 줄 단위로 이해할 때 아래 방식이 가장 효율적입니다.

1. 파일의 top-level symbol 범위를 먼저 확인한다.
2. 각 symbol 시작줄에서 함수 시그니처를 읽는다.
3. 함수 내부를 아래 순서로 본다.
- 입력 파싱
- core 계산
- 후처리
- 반환
4. 반환값이 어디서 사용되는지 역추적한다.
5. 로그 메시지 문자열을 기준으로 런타임 출력과 코드 줄을 매칭한다.

추천 도구 명령:

```bash
# 파일 라인번호 보기
nl -ba scripts/run_experiment.py | sed -n '1,220p'

# 함수 정의 빠르게 찾기
rg "^def |^class " scripts/run_ad_inference.py

# 특정 키워드 흐름 추적
rg "build_result_dict|to_llm_ad_info|generate_report" -n src scripts
```

---

## 8. 지금 상태에서 가장 먼저 해볼 학습 과제

1. `run_experiment.py`에서 `run_experiment()`만 1회 완독
2. `run_ad_inference.py`에서 `build_result_dict()` 완독
3. `src/mllm/base.py`에서 `generate_answers`, `generate_report` 완독
4. `src/eval/metrics.py`에서 `calculate_accuracy_mmad()` 완독

이 4개만 끝내면 레포 핵심 데이터 플로우는 거의 다 잡힙니다.

---

## 9. 부록 안내

아래 부록은 자동 생성 인덱스입니다.

1. 부록 A: 파일별 함수/클래스 라인 범위
2. 부록 B: 파일별 docstring 기반 API 표면

부록은 “어느 줄을 읽어야 하는지”를 빠르게 찾는 용도입니다.

---

# 부록 A. Symbol Index (Auto)

# Symbol Index (Auto)

## `apps/api/main.py` (41 lines)
- def `reports(limit)` L30-L32
- def `report_detail(report_id)` L36-L41

## `apps/dashboard/app.py` (50 lines)
- top-level symbol 없음

## `apps/dashboard/app_gradio.py` (123 lines)
- def `run_inspection(image)` L14-L46
- def `fetch_reports(limit)` L49-L70

## `scripts/eval_llm_baseline.py` (463 lines)
- def `load_mmad_data(json_path)` L65-L68
- def `load_ad_predictions(ad_output_path)` L71-L103
- def `load_ad_predictions_from_list(predictions_list)` L106-L114
- def `main()` L117-L459

## `scripts/eval_onnx.py` (312 lines)
- def `compute_auroc(scores, labels)` L31-L36
- def `compute_pixel_auroc(preds, targets)` L39-L56
- def `compute_pro(preds, targets, num_thresholds)` L59-L90
- def `load_test_data(data_root, dataset, category)` L93-L146
- def `evaluate_category(model, samples, compute_pro_metric)` L149-L205
- def `main()` L208-L306

## `scripts/export_patchcore.py` (309 lines)
- class `BackboneWrapper` L29-L60
  - def `__init__(self, feature_extractor, feature_pooler, layers)` L32-L36
  - def `forward(self, x)` L38-L60
- def `find_checkpoints(checkpoint_dir, version, datasets, categories)` L63-L113
- def `export_model(checkpoint_path, output_dir, input_size)` L116-L210
- def `main()` L213-L305

## `scripts/legacy/eval_onnx.py` (313 lines)
- def `compute_auroc(scores, labels)` L31-L36
- def `compute_pixel_auroc(preds, targets)` L39-L56
- def `compute_pro(preds, targets, num_thresholds)` L59-L90
- def `load_test_data(data_root, dataset, category)` L93-L146
- def `evaluate_category(model, samples, compute_pro_metric)` L149-L205
- def `main()` L208-L309

## `scripts/run_ad_inference.py` (1421 lines)
- def `parse_image_path(image_path)` L107-L111
- def `_is_hub_error(exc)` L114-L122
- def `_is_oom_error(exc)` L125-L126
- def `chunked(items, chunk_size)` L129-L133
- def `load_image_for_inference(image_path, decode_reduced)` L136-L157
- def `_load_image_job(job)` L160-L162
- def `compute_confidence_level(anomaly_score)` L165-L184
- def `compute_defect_location(anomaly_map, threshold)` L187-L221
- def `scale_defect_location(defect_location, from_size, to_size)` L224-L254
- def `_clip01(value)` L257-L258
- def `_deep_copy_json_obj(obj)` L261-L262
- def `load_ad_policy(policy_path)` L265-L297
- def `resolve_class_policy(policy, dataset, category)` L300-L317
- def `decide_with_policy(anomaly_score, class_policy)` L320-L328
- def `compute_location_confidence(anomaly_score, class_policy, map_stats, defect_location, review_needed)` L331-L374
- def `build_reason_codes(anomaly_score, class_policy, decision, defect_location, location_confidence)` L377-L409
- def `build_llm_guidance(decision, class_policy, reason_codes, location_confidence)` L412-L465
- class `PatchCoreCheckpointRunner` L468-L753
  - def `__init__(self, checkpoint_dir, version, threshold, device, input_size, allow_online_backbone, sync_timing, postprocess_map, use_amp)` L469-L495
  - def `_find_checkpoint(self, dataset, category)` L497-L522
  - def `list_available_models(self)` L524-L546
  - def `_load_from_checkpoint(ckpt_path, pre_trained_override)` L549-L565
  - def `_load_manual_no_hub(ckpt_path)` L568-L586
  - def `get_model(self, dataset, category)` L588-L619
  - def `_preprocess_tensor(self, image)` L621-L627
  - def `_preprocess_batch(self, images)` L629-L631
  - def `warmup_model(self, dataset, category)` L633-L644
  - def `predict(self, dataset, category, image, original_size)` L646-L654
  - def `predict_batch(self, dataset, category, images, original_sizes)` L656-L746
  - def `clear_cache(self)` L748-L753
- class `PatchCoreOnnxRunner` L756-L797
  - def `__init__(self, models_dir, threshold, device)` L757-L758
  - def `list_available_models(self)` L760-L761
  - def `warmup_model(self, dataset, category)` L763-L765
  - def `predict(self, dataset, category, image, original_size)` L767-L782
  - def `predict_batch(self, dataset, category, images, original_sizes)` L784-L794
  - def `clear_cache(self)` L796-L797
- def `load_images(mmad_json, datasets, categories)` L800-L819
- def `extract_prediction_list(payload)` L822-L829
- def `build_output_payload(results, output_format, policy, backend, model_threshold)` L832-L855
- def `save_results(path, results, output_format, policy, backend, model_threshold)` L858-L876
- def `build_result_dict(image_path, dataset, category, pred, policy, include_map_stats)` L879-L994
- def `resolve_config(args)` L997-L1017
- def `main()` L1020-L1417

## `scripts/run_experiment.py` (555 lines)
- def `stratified_sample(image_paths, n_per_folder, seed)` L50-L79
- def `load_mmad_data(json_path)` L82-L85
- def `load_ad_predictions(ad_output_path)` L88-L90
- def `run_ad_inference(cfg, data_root, mmad_json)` L93-L174
- def `resolve_paths(cfg)` L177-L195
- def `run_experiment(cfg)` L198-L441
- def `main()` L444-L551

## `scripts/test_report_pipeline.py` (94 lines)
- def `main()` L28-L90

## `scripts/train_anomalib.py` (856 lines)
- def `_patched_tqdm_init(self, *args, **kwargs)` L10-L12
- def `get_train_logger()` L48-L52
- def `get_inference_logger()` L54-L58
- class `EpochProgressCallback` L60-L83
  - def `on_train_epoch_end(self, trainer, pl_module)` L61-L83
- class `Anomalibs` L86-L829
  - def `__init__(self, config_path)` L87-L111
  - def `cleanup_memory()` L114-L122
  - def `filter_none(d)` L125-L126
  - def `get_evaluator(self)` L133-L148
  - def `get_model(self)` L150-L157
  - def `get_datamodule_kwargs(self)` L159-L168
  - def `get_engine(self, dataset, category, model, datamodule, version_dir, is_resume)` L170-L281
  - def `get_category_dir(self, dataset, category)` L288-L291
  - def `get_latest_version(self, dataset, category)` L293-L306
  - def `get_version_dir(self, dataset, category, create_new)` L308-L331
  - def `get_ckpt_path(self, dataset, category)` L333-L394
  - def `fit(self, dataset, category)` L396-L435
  - def `predict(self, dataset, category, save_json)` L437-L532
  - def `_print_metrics(self, predictions, dataset, category)` L534-L605
  - def `get_mask_path(self, image_path, dataset)` L607-L629
  - def `save_predictions_json(self, predictions, dataset, category)` L631-L675
  - def `get_all_categories(self)` L677-L690
  - def `get_trained_categories(self, filter_by_config)` L692-L738
  - def `fit_all(self)` L740-L754
  - def `predict_all(self, save_json)` L756-L829
- def `main()` L832-L852

## `scripts/upload_reports_to_pg.py` (55 lines)
- def `main()` L12-L51

## `scripts/visualize_bbox_mask_subset.py` (696 lines)
- class `SampleRecord` L56-L62
- def `normalize_rel_path(path_like)` L65-L71
- def `parse_rel_image_path(rel_path)` L74-L80
- def `extract_prediction_list(payload)` L83-L88
- def `load_predictions_index(pred_json)` L91-L102
- def `load_mmad_entries(mmad_json)` L105-L116
- def `_none_or_set(values)` L119-L122
- def `sample_records(mmad_entries, pred_index, datasets, categories, defect_types, include_good, samples_per_group, seed)` L125-L181
- def `resolve_image_path(sample, data_root)` L184-L195
- def `resolve_mask_ref(sample, data_root)` L198-L224
- def `load_mask(mask_ref, image_hw)` L227-L250
- def `sanitize_bbox(value)` L253-L259
- def `extract_pred_bbox(pred_meta)` L262-L274
- def `extract_pred_threshold(pred_meta, fallback)` L277-L282
- def `mask_to_bbox(mask)` L285-L289
- def `bbox_to_mask(shape, bbox)` L292-L305
- def `iou_binary(a, b)` L308-L315
- def `compute_bbox_mask_metrics(pred_bbox, gt_mask)` L318-L347
- def `overlay_mask(image, mask, color, alpha)` L350-L357
- def `draw_mask_contour(image, mask, color, thickness)` L360-L368
- def `draw_bbox(image, bbox, color, label)` L371-L386
- def `_fmt(v)` L389-L392
- def `add_title(panel, title)` L395-L399
- def `to_panel(image, panel_size)` L402-L403
- def `normalize_map(anomaly_map)` L406-L412
- def `build_visual_row(image, gt_mask, pred_bbox, anomaly_map, threshold, score, sample_name, metrics, panel_size)` L415-L470
- def `_mean(values)` L473-L477
- def `summarize_rows(rows)` L480-L503
- def `save_csv(path, rows)` L506-L515
- def `save_json(path, payload)` L518-L521
- def `main()` L524-L692

## `src/__init__.py` (1 lines)
- top-level symbol 없음

## `src/ad/__init__.py` (11 lines)
- top-level symbol 없음

## `src/ad/adapter.py` (82 lines)
- def `_to_bool(value)` L8-L15
- def `_to_float(value)` L18-L22
- def `to_llm_ad_info(prediction)` L25-L81

## `src/ad/io.py` (48 lines)
- def `normalize_image_key(image_key)` L10-L12
- def `_index_prediction_list(predictions)` L15-L24
- def `parse_ad_predictions_payload(payload)` L27-L41
- def `load_ad_predictions_file(path)` L44-L47

## `src/anomaly/__init__.py` (23 lines)
- top-level symbol 없음

## `src/anomaly/base.py` (172 lines)
- class `AnomalyResult` L13-L38
  - def `to_dict(self)` L29-L38
- class `BaseAnomalyModel` L41-L103
  - def `__init__(self, model_path, threshold, device, **kwargs)` L48-L65
  - def `load_model(self)` L68-L70
  - def `predict(self, image)` L73-L82
  - def `predict_batch(self, images)` L84-L96
  - def `is_loaded(self)` L98-L100
  - def `__repr__(self)` L102-L103
- class `PerClassAnomalyModel` L106-L141
  - def `__init__(self, models_dir, class_name, **kwargs)` L112-L126
  - def `list_available_classes(cls, models_dir)` L129-L141
- class `UnifiedAnomalyModel` L144-L172
  - def `__init__(self, **kwargs)` L150-L152
  - def `supported_classes(self)` L155-L157
  - def `predict_with_class(self, image, class_name)` L160-L172

## `src/anomaly/patchcore_onnx.py` (352 lines)
- class `PatchCoreOnnx` L17-L278
  - def `__init__(self, model_path, threshold, device, **kwargs)` L37-L57
  - def `load_model(self)` L59-L123
  - def `input_size(self)` L126-L128
  - def `_preprocess(self, image)` L130-L152
  - def `predict(self, image)` L154-L188
  - def `_compute_anomaly(self, features, original_size)` L190-L219
  - def `predict_batch(self, images)` L221-L223
  - def `_euclidean_dist(a, b)` L226-L231
  - def `_nearest_neighbors(self, embedding, n_neighbors)` L233-L247
  - def `_compute_image_score(self, patch_scores, locations, embedding)` L249-L278
- class `PatchCoreModelManager` L281-L352
  - def `__init__(self, models_dir, threshold, device)` L284-L300
  - def `get_model(self, dataset, category)` L302-L320
  - def `predict(self, dataset, category, image)` L322-L325
  - def `get_model_path(self, dataset, category)` L327-L329
  - def `list_available_models(self)` L331-L348
  - def `clear_cache(self)` L350-L352

## `src/common/io.py` (16 lines)
- def `read_json(path)` L7-L10
- def `imread_bgr(path)` L12-L16

## `src/common/types.py` (16 lines)
- class `MMADSample` L7-L9
- class `AnomalyResult` L12-L16

## `src/config/__init__.py` (4 lines)
- top-level symbol 없음

## `src/config/experiment.py` (80 lines)
- class `ExperimentConfig` L12-L45
  - def `experiment_name(self)` L41-L45
- def `load_experiment_config(path)` L48-L80

## `src/datasets/__init__.py` (1 lines)
- top-level symbol 없음

## `src/datasets/anomalib_folder_builder.py` (103 lines)
- def `_safe_link(src, dst, copy)` L12-L26
- class `BuiltFolderDataset` L30-L32
- def `build_anomalib_folder_dataset(train_goods, test_records, out_root, category, copy_files)` L35-L103

## `src/datasets/dataloader.py` (417 lines)
- def `iter_image_files(directory)` L20-L28
- def `collate_items(items)` L31-L55
- class `MVTecLOCODataset` L58-L163
  - def `__init__(self, root, category, split, preprocess, image_size)` L60-L74
  - def `make_dataset(self)` L76-L131
  - def `__getitem__(self, index)` L133-L163
- class `GoodsADDataset` L166-L263
  - def `__init__(self, root, category, split, preprocess, image_size)` L168-L182
  - def `make_dataset(self)` L184-L231
  - def `__getitem__(self, index)` L233-L263
- class `GoodsADDataModule` L266-L303
  - def `__init__(self, root, category, image_size, train_batch_size, eval_batch_size, num_workers, **kwargs)` L268-L279
  - def `name(self)` L282-L283
  - def `setup(self, stage)` L285-L291
  - def `train_dataloader(self)` L293-L294
  - def `val_dataloader(self)` L296-L297
  - def `test_dataloader(self)` L299-L300
  - def `predict_dataloader(self)` L302-L303
- class `MVTecLOCODataModule` L306-L353
  - def `__init__(self, root, category, image_size, train_batch_size, eval_batch_size, num_workers, **kwargs)` L308-L329
  - def `name(self)` L332-L333
  - def `setup(self, stage)` L335-L341
  - def `train_dataloader(self)` L343-L344
  - def `val_dataloader(self)` L346-L347
  - def `test_dataloader(self)` L349-L350
  - def `predict_dataloader(self)` L352-L353
- class `MMADLoader` L356-L417
  - def `__init__(self, config, model_name)` L357-L362
  - def `get_categories(self, dataset)` L364-L371
  - def `mvtec_ad(self, category, **kwargs)` L373-L374
  - def `visa(self, category, **kwargs)` L376-L377
  - def `mvtec_loco(self, category, **kwargs)` L379-L380
  - def `goods_ad(self, category, **kwargs)` L382-L383
  - def `get_datamodule(self, dataset, category, **kwargs)` L385-L410
  - def `iter_all(self, **kwargs)` L412-L417

## `src/datasets/mmad.py` (16 lines)
- def `load_mmad_samples(mmad_json_path)` L7-L9
- def `get_templates(meta, k, use_similar)` L11-L16

## `src/datasets/mmad_index_csv.py` (146 lines)
- class `MMADIndexRecord` L15-L47
  - def `is_good(self)` L31-L32
  - def `defect_type(self)` L35-L47
- def `_parse_list_cell(v)` L50-L66
- def `load_mmad_index_csv(csv_path, data_root)` L69-L106
- def `filter_by_category(records, category)` L109-L111
- def `split_good_train_test(records, train_ratio, seed)` L114-L146

## `src/datasets/preprocess.py` (5 lines)
- top-level symbol 없음

## `src/eval/__init__.py` (14 lines)
- top-level symbol 없음

## `src/eval/metrics.py` (424 lines)
- def `find_optimal_threshold(predictions, level, num_thresholds)` L33-L76
- def `compute_pro(gt_masks_np, anomaly_maps_np, num_thresholds)` L79-L130
- def `compute_anomaly_metrics(predictions, image_thresholds, pixel_thresholds, pro_num_thresholds)` L133-L207
- def `normalize_dataset_name(dataset_name)` L215-L219
- def `calculate_accuracy_mmad(answers_json_path, normal_flag, show_overkill_miss, save_csv, show_plot)` L222-L401

## `src/mllm/__init__.py` (58 lines)
- def `get_gpt4_client(*args, **kwargs)` L25-L28
- def `get_claude_client(*args, **kwargs)` L31-L34
- def `get_gemini_client(*args, **kwargs)` L37-L40
- def `get_qwen_client(*args, **kwargs)` L43-L46
- def `get_internvl_client(*args, **kwargs)` L49-L52
- def `get_llava_client(*args, **kwargs)` L55-L58

## `src/mllm/base.py` (481 lines)
- def `_parse_llm_json(text)` L109-L124
- def `_normalize_decision(value)` L127-L132
- def `format_ad_info(ad_info)` L135-L170
- def `get_mime_type(image_path)` L173-L180
- class `BaseLLMClient` L183-L481
  - def `__init__(self, max_image_size, max_retries, visualization)` L191-L200
  - def `encode_image_to_base64(self, image)` L202-L224
  - def `parse_conversation(self, meta)` L226-L259
  - def `parse_answer(self, response_text, options)` L261-L280
  - def `send_request(self, payload)` L283-L285
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L288-L305
  - def `generate_answers(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L307-L357
  - def `generate_answers_batch(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L359-L402
  - def `extract_response_text(self, response)` L405-L407
  - def `build_report_payload(self, image_path, category, ad_info)` L411-L432
  - def `generate_report(self, image_path, category, ad_info, **kwargs)` L434-L481

## `src/mllm/claude_client.py` (159 lines)
- class `ClaudeClient` L14-L159
  - def `__init__(self, api_key, model, api_url, max_tokens, **kwargs)` L17-L32
  - def `send_request(self, payload)` L34-L70
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L72-L151
  - def `extract_response_text(self, response)` L153-L159

## `src/mllm/echo.py` (66 lines)
- class `EchoMLLM` L6-L66
  - def `__init__(self, language, seed)` L13-L15
  - def `answer_mcq(self, question, structured)` L17-L38
  - def `generate_report(self, structured)` L40-L66

## `src/mllm/factory.py` (106 lines)
- def `list_llm_models()` L55-L57
- def `get_llm_client(model_name, model_path, **kwargs)` L60-L106

## `src/mllm/gemini_client.py` (208 lines)
- class `GeminiClient` L13-L208
  - def `__init__(self, api_key, model, max_retries, **kwargs)` L24-L42
  - def `_load_model(self)` L44-L57
  - def `send_request(self, payload)` L59-L79
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L81-L147
  - def `extract_response_text(self, response)` L149-L151
  - def `generate_answers(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L153-L208

## `src/mllm/internvl_client.py` (511 lines)
- def `_patch_linspace_for_meta()` L27-L43
- def `build_transform(input_size)` L49-L56
- def `find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size)` L59-L73
- def `dynamic_preprocess(image, min_num, max_num, image_size, use_thumbnail)` L76-L114
- def `load_image(image_file, input_size, max_num)` L117-L124
- class `InternVLClient` L127-L511
  - def `__init__(self, model_path, device, torch_dtype, max_new_tokens, num_gpus, max_patches, **kwargs)` L145-L164
  - def `_get_torch_dtype(self)` L166-L173
  - def `_split_model(self, model_name)` L175-L209
  - def `_load_model(self)` L211-L261
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L265-L299
  - def `send_request(self, payload)` L301-L342
  - def `extract_response_text(self, response)` L344-L346
  - def `generate_answers(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L348-L430
  - def `generate_answers_batch(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L432-L511

## `src/mllm/llava_client.py` (347 lines)
- class `LLaVAClient` L16-L347
  - def `__init__(self, model_path, device, torch_dtype, max_new_tokens, use_hf, conv_mode, temperature, **kwargs)` L30-L54
  - def `_get_torch_dtype(self)` L56-L63
  - def `_load_model_hf(self)` L65-L79
  - def `_load_model_llava(self)` L83-L98
  - def `_load_model(self)` L102-L110
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L112-L126
  - def `_generate_hf(self, payload)` L128-L184
  - def `_generate_llava(self, payload)` L186-L260
  - def `send_request(self, payload)` L262-L271
  - def `extract_response_text(self, response)` L273-L275
  - def `generate_answers(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L277-L314
  - def `generate_answers_batch(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L316-L347

## `src/mllm/openai_client.py` (151 lines)
- class `GPT4Client` L14-L151
  - def `__init__(self, api_key, model, api_url, max_tokens, **kwargs)` L19-L34
  - def `send_request(self, payload)` L36-L68
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L70-L144
  - def `extract_response_text(self, response)` L146-L151

## `src/mllm/qwen_client.py` (243 lines)
- class `QwenVLClient` L15-L243
  - def `__init__(self, model_path, device, torch_dtype, max_new_tokens, min_pixels, max_pixels, use_flash_attention, **kwargs)` L24-L45
  - def `_load_model(self)` L47-L78
  - def `build_payload(self, query_image_path, few_shot_paths, questions, ad_info, instruction)` L82-L121
  - def `send_request(self, payload)` L123-L166
  - def `extract_response_text(self, response)` L168-L170
  - def `generate_answers(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L172-L210
  - def `generate_answers_batch(self, query_image_path, meta, few_shot_paths, ad_info, instruction)` L212-L243

## `src/rag/__init__.py` (12 lines)
- top-level symbol 없음

## `src/rag/embeddings.py` (37 lines)
- def `get_embedding_model(provider, **kwargs)` L5-L37

## `src/rag/indexer.py` (90 lines)
- class `Indexer` L15-L90
  - def `__init__(self, json_path, persist_dir, embedding_provider, **embedding_kwargs)` L18-L27
  - def `load_documents(self)` L29-L58
  - def `build_index(self)` L60-L74
  - def `load_index(self)` L76-L84
  - def `get_or_create(self)` L86-L90

## `src/rag/prompt.py` (112 lines)
- def `report_prompt_rag(category, domain_knowledge)` L77-L93
- def `rag_prompt(ad_info, domain_knowledge)` L96-L112

## `src/rag/retriever.py` (84 lines)
- class `Retrievers` L7-L84
  - def `__init__(self, vectorstore)` L10-L11
  - def `retrieve(self, query, dataset, category, k)` L13-L36
  - def `build_filter(self, dataset, category)` L38-L60
  - def `format_context(self, docs)` L62-L84

## `src/report/schema.py` (10 lines)
- def `load_schema(path)` L6-L7
- def `validate_report(report, schema)` L9-L10

## `src/service/pipeline.py` (118 lines)
- def `_stem(s)` L18-L19
- class `InspectionPipeline` L22-L118
  - def `__init__(self, anomaly_model, mllm_client, runtime_cfg, pg_conn)` L25-L37
  - def `inspect(self, image_abs, dataset, category, line, templates_abs, mask_path, similar_image_path, save_to_db)` L41-L118

## `src/service/settings.py` (49 lines)
- def `_expand_env(text)` L10-L15
- def `load_yaml(path)` L17-L20
- class `RuntimePaths` L23-L27
- class `RuntimeConfig` L30-L34
- def `load_runtime_config(path)` L36-L49

## `src/storage/db.py` (71 lines)
- def `connect(db_path)` L25-L30
- def `insert_report(conn, report)` L32-L48
- def `list_reports(conn, limit)` L50-L57
- def `get_report(conn, report_id)` L59-L71

## `src/storage/pg.py` (109 lines)
- def `connect(dsn)` L36-L40
- def `create_tables(conn)` L43-L47
- def `insert_report(conn, data)` L50-L85
- def `list_reports(conn, limit)` L88-L97
- def `get_report(conn, report_id)` L100-L109

## `src/structure/defect.py` (159 lines)
- def `nine_grid_location(cx, cy)` L12-L27
- def `structure_from_heatmap(heatmap, thr)` L30-L123
- def `get_defect_description(structure)` L126-L159

## `src/structure/render.py` (19 lines)
- def `save_heatmap_and_overlay(image_bgr, heatmap, out_dir, stem)` L6-L19

## `src/utils/__init__.py` (20 lines)
- top-level symbol 없음

## `src/utils/checkpoint.py` (25 lines)
- def `save_checkpoint(model, path)` L9-L13
- def `load_checkpoint(model, path, device)` L15-L25

## `src/utils/device.py` (15 lines)
- def `get_device(verbose)` L3-L15

## `src/utils/loaders.py` (34 lines)
- def `load_env()` L12-L13
- def `load_config(config_path)` L15-L24
- def `load_json(json_path)` L26-L30
- def `load_csv(csv_path)` L32-L34

## `src/utils/log.py` (73 lines)
- def `setup_logger(name, log_dir, log_level, log_prefix, file_logging, console_logging)` L11-L73

## `src/utils/path.py` (42 lines)
- def `get_project_root()` L5-L17
- def `get_logs_dir()` L20-L24
- def `get_checkpoints_dir()` L27-L31
- def `get_output_dir()` L34-L38

## `src/utils/wandbs.py` (72 lines)
- def `login_wandb()` L9-L16
- def `init_wandb(project, name, config, tags)` L19-L28
- def `log_metrics(metrics, step)` L31-L35
- def `log_summary(metrics)` L38-L46
- def `log_images(images, key)` L49-L66
- def `finish_wandb()` L69-L72

## `src/visual/__init__.py` (18 lines)
- top-level symbol 없음

## `src/visual/comparison.py` (230 lines)
- def `create_comparison_image(original, gt_mask, pred_heatmap, anomaly_score, threshold, is_anomaly_gt, defect_type)` L15-L133
- def `create_grid_visualization(images, max_cols, spacing)` L136-L193
- def `save_comparison(output_path, original, gt_mask, pred_heatmap, anomaly_score, threshold, is_anomaly_gt, defect_type)` L196-L230

## `src/visual/plot.py` (659 lines)
- def `set_korean_font(verbose)` L14-L49
- def `count_plot(df, col, ax, figsize, palette, rotation, title, xlabel, ylabel, order, orient, top_n, show)` L52-L92
- def `bar_plot(df, x_col, y_col, ax, figsize, hue, palette, rotation, title, xlabel, ylabel, show)` L95-L114
- def `line_plot(df, x_col, y_col, ax, figsize, color, marker, linewidth, rotation, title, show)` L117-L133
- def `box_plot(df, col, hue, ax, figsize, palette, title, show)` L136-L161
- def `hist_plot(df, col, ax, figsize, bins, color, kde, stat, title, xlabel, ylabel, show)` L164-L184
- def `kde_plot(df, col, ax, figsize, color, fill, alpha, linewidth, hue, palette, title, xlabel, ylabel, show)` L187-L211
- def `heatmap_plot(data, ax, figsize, cmap, annot, fmt, linewidths, cbar, title, xlabel, ylabel, rotation_x, rotation_y, show)` L214-L245
- def `img_plot(path, ax, figsize, title, show)` L248-L262
- def `tensor_to_numpy(tensor)` L267-L281
- def `load_original_image(image_path, target_size)` L284-L289
- def `visualize_anomaly_prediction(image_path, anomaly_map, pred_mask, gt_mask, pred_score, pred_label, figsize, cmap, alpha, title, show)` L292-L408
- def `visualize_predictions_from_runner(runner, n_samples_per_category, filter_by, random_sample, categories, show_inference_time, figsize, show)` L411-L551
- def `visualize_single_prediction(model_name, dataset, category, sample_idx, config_path, figsize, show)` L554-L656

# 부록 B. API Surface (Auto)

# API Surface (Auto)

## `apps/api/main.py`
- module doc: (none)
- def `reports` L30-L32
  - doc: 최근 N개 리포트 목록 조회.
- def `report_detail` L36-L41
  - doc: 리포트 단건 조회.

## `apps/dashboard/app.py`
- module doc: (none)
- symbols: 없음

## `apps/dashboard/app_gradio.py`
- module doc: (none)
- def `run_inspection` L14-L46
  - doc: 이미지 검사 실행
- def `fetch_reports` L49-L70
  - doc: 최신 리포트 조회

## `scripts/eval_llm_baseline.py`
- module doc: MMAD LLM Baseline Evaluation Script
- def `load_mmad_data` L65-L68
  - doc: Load MMAD dataset JSON.
- def `load_ad_predictions` L71-L103
  - doc: Load anomaly detection model predictions.
- def `load_ad_predictions_from_list` L106-L114
  - doc: Helper to index predictions list by image path.
- def `main` L117-L459

## `scripts/eval_onnx.py`
- module doc: Evaluate PatchCore ONNX models on test set.
- def `compute_auroc` L31-L36
  - doc: Compute AUROC from scores and binary labels.
- def `compute_pixel_auroc` L39-L56
  - doc: Compute pixel-level AUROC.
- def `compute_pro` L59-L90
  - doc: Compute Per-Region Overlap (PRO) score.
- def `load_test_data` L93-L146
  - doc: Load test images and ground truth.
- def `evaluate_category` L149-L205
  - doc: Evaluate model on samples.
- def `main` L208-L306

## `scripts/export_patchcore.py`
- module doc: Export PatchCore models for optimized inference.
- class `BackboneWrapper` L29-L60
  - doc: Wrapper that extracts and concatenates features from backbone.
  - method `__init__` L32-L36
  - method `forward` L38-L60
    - doc: Extract and concatenate features from specified layers.
- def `find_checkpoints` L63-L113
  - doc: Find PatchCore checkpoints.
- def `export_model` L116-L210
  - doc: Export PatchCore model: backbone to ONNX + memory bank to numpy.
- def `main` L213-L305

## `scripts/legacy/eval_onnx.py`
- module doc: Evaluate PatchCore ONNX models on test set.
- def `compute_auroc` L31-L36
  - doc: Compute AUROC from scores and binary labels.
- def `compute_pixel_auroc` L39-L56
  - doc: Compute pixel-level AUROC.
- def `compute_pro` L59-L90
  - doc: Compute Per-Region Overlap (PRO) score.
- def `load_test_data` L93-L146
  - doc: Load test images and ground truth.
- def `evaluate_category` L149-L205
  - doc: Evaluate model on samples.
- def `main` L208-L309

## `scripts/run_ad_inference.py`
- module doc: Unified PatchCore inference runner (checkpoint + ONNX backends).
- def `parse_image_path` L107-L111
- def `_is_hub_error` L114-L122
- def `_is_oom_error` L125-L126
- def `chunked` L129-L133
- def `load_image_for_inference` L136-L157
- def `_load_image_job` L160-L162
- def `compute_confidence_level` L165-L184
- def `compute_defect_location` L187-L221
- def `scale_defect_location` L224-L254
  - doc: Scale bbox coordinates from map size to original image size.
- def `_clip01` L257-L258
- def `_deep_copy_json_obj` L261-L262
- def `load_ad_policy` L265-L297
- def `resolve_class_policy` L300-L317
- def `decide_with_policy` L320-L328
- def `compute_location_confidence` L331-L374
- def `build_reason_codes` L377-L409
- def `build_llm_guidance` L412-L465
- class `PatchCoreCheckpointRunner` L468-L753
  - method `__init__` L469-L495
  - method `_find_checkpoint` L497-L522
  - method `list_available_models` L524-L546
  - method `_load_from_checkpoint` L549-L565
  - method `_load_manual_no_hub` L568-L586
  - method `get_model` L588-L619
  - method `_preprocess_tensor` L621-L627
  - method `_preprocess_batch` L629-L631
  - method `warmup_model` L633-L644
  - method `predict` L646-L654
  - method `predict_batch` L656-L746
  - method `clear_cache` L748-L753
- class `PatchCoreOnnxRunner` L756-L797
  - method `__init__` L757-L758
  - method `list_available_models` L760-L761
  - method `warmup_model` L763-L765
  - method `predict` L767-L782
  - method `predict_batch` L784-L794
  - method `clear_cache` L796-L797
- def `load_images` L800-L819
- def `extract_prediction_list` L822-L829
- def `build_output_payload` L832-L855
- def `save_results` L858-L876
- def `build_result_dict` L879-L994
- def `resolve_config` L997-L1017
- def `main` L1020-L1417

## `scripts/run_experiment.py`
- module doc: Experiment Runner — run MMAD evaluation with YAML config + CLI overrides.
- def `stratified_sample` L50-L79
  - doc: 폴더(dataset/category/split)별 N장 샘플링.
- def `load_mmad_data` L82-L85
  - doc: Load MMAD dataset JSON.
- def `load_ad_predictions` L88-L90
  - doc: Load AD predictions JSON and index by normalized image path.
- def `run_ad_inference` L93-L174
  - doc: Run AD model inference using run_ad_inference.py.
- def `resolve_paths` L177-L195
  - doc: Resolve data_root and mmad_json from config, env vars, or defaults.
- def `run_experiment` L198-L441
  - doc: Run a single experiment and return the answers JSON path.
- def `main` L444-L551

## `scripts/test_report_pipeline.py`
- module doc: Single-image test: InternVL report generation → PostgreSQL storage.
- def `main` L28-L90

## `scripts/train_anomalib.py`
- module doc: (none)
- def `_patched_tqdm_init` L10-L12
- def `get_train_logger` L48-L52
- def `get_inference_logger` L54-L58
- class `EpochProgressCallback` L60-L83
  - method `on_train_epoch_end` L61-L83
- class `Anomalibs` L86-L829
  - method `__init__` L87-L111
  - method `cleanup_memory` L114-L122
    - doc: GPU 및 시스템 메모리 캐시 강제 비활성화 및 정리
  - method `filter_none` L125-L126
  - method `get_evaluator` L133-L148
    - doc: val_metrics 포함 Evaluator 생성 (validation 시 메트릭 로깅용)
  - method `get_model` L150-L157
  - method `get_datamodule_kwargs` L159-L168
  - method `get_engine` L170-L281
    - doc: 학습(fit) 전용 Engine 생성. predict는 Engine을 사용하지 않음.
  - method `get_category_dir` L288-L291
    - doc: 카테고리 디렉토리 경로 반환.
  - method `get_latest_version` L293-L306
    - doc: 가장 최신 버전 번호 반환. 없으면 None.
  - method `get_version_dir` L308-L331
    - doc: 버전 디렉토리 경로 반환.
  - method `get_ckpt_path` L333-L394
    - doc: 체크포인트 경로 반환.
  - method `fit` L396-L435
  - method `predict` L437-L532
  - method `_print_metrics` L534-L605
    - doc: Compute and print evaluation metrics.
  - method `get_mask_path` L607-L629
    - doc: 이미지 경로에서 대응하는 마스크 경로 추론
  - method `save_predictions_json` L631-L675
  - method `get_all_categories` L677-L690
    - doc: Get list of (dataset, category) tuples from DATASETS.
  - method `get_trained_categories` L692-L738
    - doc: Get list of (dataset, category) tuples that have trained checkpoints.
  - method `fit_all` L740-L754
  - method `predict_all` L756-L829
- def `main` L832-L852

## `scripts/upload_reports_to_pg.py`
- module doc: Read JSON report file(s) and insert into PostgreSQL.
- def `main` L12-L51

## `scripts/visualize_bbox_mask_subset.py`
- module doc: Visualize bbox-vs-mask quality on sampled MMAD images.
- class `SampleRecord` L56-L62
- def `normalize_rel_path` L65-L71
- def `parse_rel_image_path` L74-L80
- def `extract_prediction_list` L83-L88
- def `load_predictions_index` L91-L102
- def `load_mmad_entries` L105-L116
- def `_none_or_set` L119-L122
- def `sample_records` L125-L181
- def `resolve_image_path` L184-L195
- def `resolve_mask_ref` L198-L224
- def `load_mask` L227-L250
- def `sanitize_bbox` L253-L259
- def `extract_pred_bbox` L262-L274
- def `extract_pred_threshold` L277-L282
- def `mask_to_bbox` L285-L289
- def `bbox_to_mask` L292-L305
- def `iou_binary` L308-L315
- def `compute_bbox_mask_metrics` L318-L347
- def `overlay_mask` L350-L357
- def `draw_mask_contour` L360-L368
- def `draw_bbox` L371-L386
- def `_fmt` L389-L392
- def `add_title` L395-L399
- def `to_panel` L402-L403
- def `normalize_map` L406-L412
- def `build_visual_row` L415-L470
- def `_mean` L473-L477
- def `summarize_rows` L480-L503
- def `save_csv` L506-L515
- def `save_json` L518-L521
- def `main` L524-L692

## `src/__init__.py`
- module doc: (none)
- symbols: 없음

## `src/ad/__init__.py`
- module doc: AD integration helpers for evaluation/service pipelines.
- symbols: 없음

## `src/ad/adapter.py`
- module doc: Adapters between AD outputs and LLM-friendly input schema.
- def `_to_bool` L8-L15
- def `_to_float` L18-L22
- def `to_llm_ad_info` L25-L81
  - doc: Convert AD prediction payload to the shape expected by LLM prompts.

## `src/ad/io.py`
- module doc: AD prediction JSON I/O helpers.
- def `normalize_image_key` L10-L12
  - doc: Normalize keys for robust image-path matching across platforms.
- def `_index_prediction_list` L15-L24
- def `parse_ad_predictions_payload` L27-L41
  - doc: Parse legacy/report-style AD outputs into a normalized dict index.
- def `load_ad_predictions_file` L44-L47

## `src/anomaly/__init__.py`
- module doc: Anomaly detection module.
- symbols: 없음

## `src/anomaly/base.py`
- module doc: Base class for anomaly detection models.
- class `AnomalyResult` L13-L38
  - doc: Anomaly detection result.
  - method `to_dict` L29-L38
    - doc: Convert to dictionary for JSON serialization.
- class `BaseAnomalyModel` L41-L103
  - doc: Abstract base class for anomaly detection models.
  - method `__init__` L48-L65
    - doc: Initialize anomaly detection model.
  - method `load_model` L68-L70
    - doc: Load model from model_path. Must be implemented by subclass.
  - method `predict` L73-L82
    - doc: Run inference on a single image.
  - method `predict_batch` L84-L96
    - doc: Run inference on multiple images.
  - method `is_loaded` L98-L100
    - doc: Check if model is loaded.
  - method `__repr__` L102-L103
- class `PerClassAnomalyModel` L106-L141
  - doc: Base class for per-class anomaly models (EfficientAD, PatchCore).
  - method `__init__` L112-L126
    - doc: Initialize per-class anomaly model.
  - method `list_available_classes` L129-L141
    - doc: List available classes in the models directory.
- class `UnifiedAnomalyModel` L144-L172
  - doc: Base class for unified anomaly models (UniAD).
  - method `__init__` L150-L152
  - method `supported_classes` L155-L157
    - doc: List of classes supported by this model.
  - method `predict_with_class` L160-L172
    - doc: Run inference with class information.

## `src/anomaly/patchcore_onnx.py`
- module doc: PatchCore ONNX inference module.
- class `PatchCoreOnnx` L17-L278
  - doc: PatchCore ONNX inference class.
  - method `__init__` L37-L57
    - doc: Initialize PatchCore ONNX model.
  - method `load_model` L59-L123
    - doc: Load backbone ONNX and memory bank.
  - method `input_size` L126-L128
    - doc: Get model input size.
  - method `_preprocess` L130-L152
    - doc: Preprocess image for model input.
  - method `predict` L154-L188
    - doc: Run inference on a single image.
  - method `_compute_anomaly` L190-L219
    - doc: Compute anomaly map and score from features and memory bank.
  - method `predict_batch` L221-L223
    - doc: Run inference on multiple images.
  - method `_euclidean_dist` L226-L231
    - doc: Compute pairwise L2 distances between rows of a and b.
  - method `_nearest_neighbors` L233-L247
    - doc: Find nearest neighbors in the memory bank.
  - method `_compute_image_score` L249-L278
    - doc: Compute image-level score using anomalib PatchCore weighting.
- class `PatchCoreModelManager` L281-L352
  - doc: Manager for multiple PatchCore ONNX models.
  - method `__init__` L284-L300
    - doc: Initialize model manager.
  - method `get_model` L302-L320
    - doc: Get or load model for dataset/category.
  - method `predict` L322-L325
    - doc: Run inference for specific dataset/category.
  - method `get_model_path` L327-L329
    - doc: Get model directory path for dataset/category.
  - method `list_available_models` L331-L348
    - doc: List available dataset/category pairs.
  - method `clear_cache` L350-L352
    - doc: Clear loaded models from cache.

## `src/common/io.py`
- module doc: (none)
- def `read_json` L7-L10
- def `imread_bgr` L12-L16

## `src/common/types.py`
- module doc: (none)
- class `MMADSample` L7-L9
- class `AnomalyResult` L12-L16

## `src/config/__init__.py`
- module doc: Experiment configuration utilities.
- symbols: 없음

## `src/config/experiment.py`
- module doc: Experiment configuration loader.
- class `ExperimentConfig` L12-L45
  - doc: Configuration for a single experiment run.
  - method `experiment_name` L41-L45
    - doc: Auto-generate experiment name: {ad_model}_{llm}_{few_shot}shot[_rag]
- def `load_experiment_config` L48-L80
  - doc: Load ExperimentConfig from a YAML file.

## `src/datasets/__init__.py`
- module doc: (none)
- symbols: 없음

## `src/datasets/anomalib_folder_builder.py`
- module doc: (none)
- def `_safe_link` L12-L26
  - doc: Create a symlink (preferred) or copy if requested / unsupported.
- class `BuiltFolderDataset` L30-L32
- def `build_anomalib_folder_dataset` L35-L103
  - doc: Build an anomalib Folder-format dataset from MMAD_index.csv records.

## `src/datasets/dataloader.py`
- module doc: (none)
- def `iter_image_files` L20-L28
  - doc: Yield image files in a directory with extension-insensitive filtering.
- def `collate_items` L31-L55
  - doc: ImageItem 리스트를 ImageBatch로 변환하는 collate 함수.
- class `MVTecLOCODataset` L58-L163
  - doc: MVTec-LOCO 커스텀 Dataset (중첩 마스크 구조 처리)
  - method `__init__` L60-L74
  - method `make_dataset` L76-L131
    - doc: MVTec-LOCO 샘플 DataFrame 생성
  - method `__getitem__` L133-L163
- class `GoodsADDataset` L166-L263
  - doc: GoodsAD 커스텀 Dataset (이미지 jpg, 마스크 png 처리)
  - method `__init__` L168-L182
  - method `make_dataset` L184-L231
    - doc: GoodsAD 샘플 DataFrame 생성
  - method `__getitem__` L233-L263
- class `GoodsADDataModule` L266-L303
  - doc: GoodsAD 커스텀 DataModule
  - method `__init__` L268-L279
  - method `name` L282-L283
  - method `setup` L285-L291
  - method `train_dataloader` L293-L294
  - method `val_dataloader` L296-L297
  - method `test_dataloader` L299-L300
  - method `predict_dataloader` L302-L303
- class `MVTecLOCODataModule` L306-L353
  - doc: MVTec-LOCO 커스텀 DataModule
  - method `__init__` L308-L329
  - method `name` L332-L333
  - method `setup` L335-L341
  - method `train_dataloader` L343-L344
  - method `val_dataloader` L346-L347
  - method `test_dataloader` L349-L350
  - method `predict_dataloader` L352-L353
- class `MMADLoader` L356-L417
  - method `__init__` L357-L362
  - method `get_categories` L364-L371
  - method `mvtec_ad` L373-L374
  - method `visa` L376-L377
  - method `mvtec_loco` L379-L380
  - method `goods_ad` L382-L383
  - method `get_datamodule` L385-L410
    - doc: DataModule 생성. config에서 기본값 로드 후, kwargs로 오버라이드 가능.
  - method `iter_all` L412-L417

## `src/datasets/mmad.py`
- module doc: (none)
- def `load_mmad_samples` L7-L9
- def `get_templates` L11-L16

## `src/datasets/mmad_index_csv.py`
- module doc: (none)
- class `MMADIndexRecord` L15-L47
  - doc: A single record parsed from MMAD_index.csv.
  - method `is_good` L31-L32
  - method `defect_type` L35-L47
    - doc: Defect type name inferred from image_path.
- def `_parse_list_cell` L50-L66
- def `load_mmad_index_csv` L69-L106
  - doc: Load MMAD_index.csv and resolve relative paths using `data_root`.
- def `filter_by_category` L109-L111
- def `split_good_train_test` L114-L146
  - doc: Split *good* samples into train/test while keeping all bad samples in test.

## `src/datasets/preprocess.py`
- module doc: (none)
- symbols: 없음

## `src/eval/__init__.py`
- module doc: Evaluation modules for MMAD.
- symbols: 없음

## `src/eval/metrics.py`
- module doc: Evaluation Metrics for Anomaly Detection.
- def `find_optimal_threshold` L33-L76
  - doc: 카테고리별 F1 최대화 기반 optimal threshold 탐색.
- def `compute_pro` L79-L130
  - doc: Per-Region Overlap (PRO) 계산.
- def `compute_anomaly_metrics` L133-L207
  - doc: Anomaly detection 전체 지표 산출.
- def `normalize_dataset_name` L215-L219
  - doc: Normalize dataset names (merge DS-MVTec and MVTec-AD).
- def `calculate_accuracy_mmad` L222-L401
  - doc: Calculate MMAD evaluation metrics - matches paper's caculate_accuracy_mmad().

## `src/mllm/__init__.py`
- module doc: MLLM clients for MMAD evaluation.
- def `get_gpt4_client` L25-L28
  - doc: Get GPT-4o/GPT-4V client.
- def `get_claude_client` L31-L34
  - doc: Get Claude client.
- def `get_gemini_client` L37-L40
  - doc: Get Gemini client (FREE tier available!).
- def `get_qwen_client` L43-L46
  - doc: Get Qwen2.5-VL client (requires transformers, qwen-vl-utils).
- def `get_internvl_client` L49-L52
  - doc: Get InternVL2 client (requires transformers).
- def `get_llava_client` L55-L58
  - doc: Get LLaVA client (requires transformers or llava package).

## `src/mllm/base.py`
- module doc: Base class for LLM clients following MMAD paper evaluation protocol.
- def `_parse_llm_json` L109-L124
  - doc: Extract and parse JSON from LLM response text.
- def `_normalize_decision` L127-L132
  - doc: Normalize various LLM is_anomaly outputs to bool.
- def `format_ad_info` L135-L170
  - doc: Format AD model output as a concise natural language summary.
- def `get_mime_type` L173-L180
  - doc: Get MIME type from image path.
- class `BaseLLMClient` L183-L481
  - doc: Base class for MMAD LLM evaluation.
  - method `__init__` L191-L200
  - method `encode_image_to_base64` L202-L224
    - doc: Encode image to base64, resizing if necessary.
  - method `parse_conversation` L226-L259
    - doc: Parse MMAD conversation format into questions, answers, and types.
  - method `parse_answer` L261-L280
    - doc: Parse answer letters from LLM response.
  - method `send_request` L283-L285
    - doc: Send request to LLM API. Must be implemented by subclass.
  - method `build_payload` L288-L305
    - doc: Build API payload. Must be implemented by subclass.
  - method `generate_answers` L307-L357
    - doc: Generate answers for all questions in the conversation.
  - method `generate_answers_batch` L359-L402
    - doc: Generate answers for all questions in a single API call.
  - method `extract_response_text` L405-L407
    - doc: Extract text content from API response. Must be implemented by subclass.
  - method `build_report_payload` L411-L432
    - doc: Build payload for report generation.
  - method `generate_report` L434-L481
    - doc: Generate a structured inspection report for a single image.

## `src/mllm/claude_client.py`
- module doc: Anthropic Claude client for MMAD evaluation.
- class `ClaudeClient` L14-L159
  - doc: Claude client following MMAD paper's evaluation protocol.
  - method `__init__` L17-L32
  - method `send_request` L34-L70
    - doc: Send request to Anthropic API with retry logic.
  - method `build_payload` L72-L151
    - doc: Build Anthropic API payload following MMAD protocol.
  - method `extract_response_text` L153-L159
    - doc: Extract text from Claude response.

## `src/mllm/echo.py`
- module doc: (none)
- class `EchoMLLM` L6-L66
  - doc: Step 1 placeholder.
  - method `__init__` L13-L15
  - method `answer_mcq` L17-L38
  - method `generate_report` L40-L66

## `src/mllm/factory.py`
- module doc: LLM client factory — shared registry and instantiation logic.
- def `list_llm_models` L55-L57
  - doc: Return sorted list of available model names.
- def `get_llm_client` L60-L106
  - doc: Factory function to get LLM client by name.

## `src/mllm/gemini_client.py`
- module doc: Google Gemini client for MMAD evaluation.
- class `GeminiClient` L13-L208
  - doc: Gemini client following MMAD paper's evaluation protocol.
  - method `__init__` L24-L42
  - method `_load_model` L44-L57
    - doc: Lazy load Gemini model.
  - method `send_request` L59-L79
    - doc: Send request to Gemini API with retry logic.
  - method `build_payload` L81-L147
    - doc: Build Gemini API payload following paper's format.
  - method `extract_response_text` L149-L151
    - doc: Extract text from Gemini response.
  - method `generate_answers` L153-L208
    - doc: Generate answers following paper's Gemini approach.

## `src/mllm/internvl_client.py`
- module doc: InternVL client for MMAD evaluation - HuggingFace transformers.
- def `_patch_linspace_for_meta` L27-L43
  - doc: Workaround: InternVL calls .item() on torch.linspace() during __init__.
- def `build_transform` L49-L56
  - doc: Build image transform for InternVL.
- def `find_closest_aspect_ratio` L59-L73
  - doc: Find closest aspect ratio for dynamic preprocessing.
- def `dynamic_preprocess` L76-L114
  - doc: Dynamic preprocessing for InternVL2.
- def `load_image` L117-L124
  - doc: Load and preprocess image for InternVL.
- class `InternVLClient` L127-L511
  - doc: InternVL client using HuggingFace transformers.
  - method `__init__` L145-L164
  - method `_get_torch_dtype` L166-L173
    - doc: Convert string dtype to torch dtype.
  - method `_split_model` L175-L209
    - doc: Create device map for multi-GPU inference.
  - method `_load_model` L211-L261
    - doc: Lazy load model and tokenizer.
  - method `build_payload` L265-L299
    - doc: Build InternVL message format.
  - method `send_request` L301-L342
    - doc: Process request using local model.
  - method `extract_response_text` L344-L346
    - doc: Extract text from response.
  - method `generate_answers` L348-L430
    - doc: Generate answers with conversation history (InternVL's approach).
  - method `generate_answers_batch` L432-L511
    - doc: Generate answers for ALL questions in a single model call (5-8x faster).

## `src/mllm/llava_client.py`
- module doc: LLaVA client for MMAD evaluation - using llava package or transformers.
- class `LLaVAClient` L16-L347
  - doc: LLaVA client for MMAD evaluation.
  - method `__init__` L30-L54
  - method `_get_torch_dtype` L56-L63
    - doc: Convert string dtype to torch dtype.
  - method `_load_model_hf` L65-L79
    - doc: Load model using HuggingFace transformers (recommended).
  - method `_load_model_llava` L83-L98
    - doc: Load model using original llava package.
  - method `_load_model` L102-L110
    - doc: Lazy load model.
  - method `build_payload` L112-L126
    - doc: Build LLaVA message format.
  - method `_generate_hf` L128-L184
    - doc: Generate response using HuggingFace transformers.
  - method `_generate_llava` L186-L260
    - doc: Generate response using original llava package.
  - method `send_request` L262-L271
    - doc: Process request using local model.
  - method `extract_response_text` L273-L275
    - doc: Extract text from response.
  - method `generate_answers` L277-L314
    - doc: Generate answers one question at a time.
  - method `generate_answers_batch` L316-L347
    - doc: Generate answers for ALL questions in a single model call (5-8x faster).

## `src/mllm/openai_client.py`
- module doc: OpenAI GPT-4o client for MMAD evaluation - matches paper's implementation.
- class `GPT4Client` L14-L151
  - doc: GPT-4o/GPT-4V client following MMAD paper's exact implementation.
  - method `__init__` L19-L34
  - method `send_request` L36-L68
    - doc: Send request to OpenAI API with retry logic.
  - method `build_payload` L70-L144
    - doc: Build OpenAI API payload following paper's format.
  - method `extract_response_text` L146-L151
    - doc: Extract text from OpenAI response.

## `src/mllm/qwen_client.py`
- module doc: Qwen2.5-VL client for MMAD evaluation - HuggingFace transformers.
- class `QwenVLClient` L15-L243
  - doc: Qwen2.5-VL client using HuggingFace transformers.
  - method `__init__` L24-L45
  - method `_load_model` L47-L78
    - doc: Lazy load model and processor.
  - method `build_payload` L82-L121
    - doc: Build Qwen VL message format.
  - method `send_request` L123-L166
    - doc: Process request using local model.
  - method `extract_response_text` L168-L170
    - doc: Extract text from response.
  - method `generate_answers` L172-L210
    - doc: Generate answers one question at a time (Qwen's approach).
  - method `generate_answers_batch` L212-L243
    - doc: Generate answers for ALL questions in a single model call (5-8x faster).

## `src/rag/__init__.py`
- module doc: (none)
- symbols: 없음

## `src/rag/embeddings.py`
- module doc: (none)
- def `get_embedding_model` L5-L37
  - doc: Return a LangChain-compatible embedding model.

## `src/rag/indexer.py`
- module doc: (none)
- class `Indexer` L15-L90
  - doc: Load domain_knowledge.json and build / load a Chroma vector store.
  - method `__init__` L18-L27
  - method `load_documents` L29-L58
    - doc: Parse domain_knowledge.json into LangChain Documents.
  - method `build_index` L60-L74
    - doc: Build a new Chroma vector store from the JSON and persist it.
  - method `load_index` L76-L84
    - doc: Load an existing persisted Chroma vector store.
  - method `get_or_create` L86-L90
    - doc: Load existing index if available, otherwise build a new one.

## `src/rag/prompt.py`
- module doc: (none)
- def `report_prompt_rag` L77-L93
  - doc: Build a RAG-augmented report generation prompt.
- def `rag_prompt` L96-L112
  - doc: Build a complete RAG-augmented instruction prompt.

## `src/rag/retriever.py`
- module doc: (none)
- class `Retrievers` L7-L84
  - doc: Semantic search over domain knowledge with optional metadata filtering.
  - method `__init__` L10-L11
  - method `retrieve` L13-L36
    - doc: Search for relevant domain knowledge documents.
  - method `build_filter` L38-L60
    - doc: Build a Chroma metadata filter dict.
  - method `format_context` L62-L84
    - doc: Format retrieved documents into a text block for MLLM prompts.

## `src/report/schema.py`
- module doc: (none)
- def `load_schema` L6-L7
- def `validate_report` L9-L10

## `src/service/pipeline.py`
- module doc: (none)
- def `_stem` L18-L19
- class `InspectionPipeline` L22-L118
  - doc: End-to-end pipeline: AD inference -> LLM report -> PostgreSQL storage.
  - method `__init__` L25-L37
  - method `inspect` L41-L118
    - doc: Run full inspection and optionally store the result in PostgreSQL.

## `src/service/settings.py`
- module doc: (none)
- def `_expand_env` L10-L15
- def `load_yaml` L17-L20
- class `RuntimePaths` L23-L27
- class `RuntimeConfig` L30-L34
- def `load_runtime_config` L36-L49

## `src/storage/db.py`
- module doc: (none)
- def `connect` L25-L30
- def `insert_report` L32-L48
- def `list_reports` L50-L57
- def `get_report` L59-L71

## `src/storage/pg.py`
- module doc: PostgreSQL CRUD for inspection_reports table.
- def `connect` L36-L40
  - doc: Connect to PostgreSQL and ensure tables exist.
- def `create_tables` L43-L47
  - doc: Create inspection_reports table if not exists.
- def `insert_report` L50-L85
  - doc: Insert a report row and return its id.
- def `list_reports` L88-L97
  - doc: Return the most recent N reports.
- def `get_report` L100-L109
  - doc: Return a single report by id, or None.

## `src/structure/defect.py`
- module doc: Defect structure analysis from anomaly heatmaps.
- def `nine_grid_location` L12-L27
  - doc: Get location description in 3x3 grid.
- def `structure_from_heatmap` L30-L123
  - doc: Extract structured defect information from anomaly heatmap.
- def `get_defect_description` L126-L159
  - doc: Generate human-readable description from defect structure.

## `src/structure/render.py`
- module doc: (none)
- def `save_heatmap_and_overlay` L6-L19

## `src/utils/__init__.py`
- module doc: (none)
- symbols: 없음

## `src/utils/checkpoint.py`
- module doc: (none)
- def `save_checkpoint` L9-L13
- def `load_checkpoint` L15-L25

## `src/utils/device.py`
- module doc: (none)
- def `get_device` L3-L15
  - doc: CUDA, MPS, CPU

## `src/utils/loaders.py`
- module doc: (none)
- def `load_env` L12-L13
- def `load_config` L15-L24
  - doc: YAML 설정 파일 로드
- def `load_json` L26-L30
  - doc: JSON 파일 로드
- def `load_csv` L32-L34
  - doc: CSV 파일 로드

## `src/utils/log.py`
- module doc: (none)
- def `setup_logger` L11-L73
  - doc: Args:

## `src/utils/path.py`
- module doc: (none)
- def `get_project_root` L5-L17
  - doc: 프로젝트 루트 경로 반환.
- def `get_logs_dir` L20-L24
  - doc: 로그 디렉토리 경로 반환 (없으면 생성).
- def `get_checkpoints_dir` L27-L31
  - doc: 체크포인트 디렉토리 경로 반환 (없으면 생성).
- def `get_output_dir` L34-L38
  - doc: 출력 디렉토리 경로 반환 (없으면 생성).

## `src/utils/wandbs.py`
- module doc: (none)
- def `login_wandb` L9-L16
- def `init_wandb` L19-L28
- def `log_metrics` L31-L35
- def `log_summary` L38-L46
- def `log_images` L49-L66
  - doc: W&B에 이미지 로깅
- def `finish_wandb` L69-L72

## `src/visual/__init__.py`
- module doc: (none)
- symbols: 없음

## `src/visual/comparison.py`
- module doc: Visualization module for comparing GT masks with model predictions.
- def `create_comparison_image` L15-L133
  - doc: Create a side-by-side comparison image.
- def `create_grid_visualization` L136-L193
  - doc: Create a grid visualization from multiple images.
- def `save_comparison` L196-L230
  - doc: Save comparison image to file.

## `src/visual/plot.py`
- module doc: (none)
- def `set_korean_font` L14-L49
  - doc: 한글 폰트 설정. Mac, Windows, Linux 지원. verbose=True면 메시지 출력.
- def `count_plot` L52-L92
  - doc: order: 'desc'(내림차순), 'asc'(오름차순), None(정렬 안 함)
- def `bar_plot` L95-L114
- def `line_plot` L117-L133
- def `box_plot` L136-L161
  - doc: 단일 boxplot 또는 hue로 그룹 비교
- def `hist_plot` L164-L184
  - doc: 히스토그램 시각화
- def `kde_plot` L187-L211
  - doc: KDE(커널 밀도 추정) 시각화
- def `heatmap_plot` L214-L245
  - doc: 히트맵 시각화 (crosstab, pivot table, correlation matrix 등에 사용)
- def `img_plot` L248-L262
  - doc: 이미지 파일 시각화. subplot 호환.
- def `tensor_to_numpy` L267-L281
  - doc: Tensor를 numpy 배열로 변환
- def `load_original_image` L284-L289
  - doc: 원본 이미지를 직접 로드 (정규화 없이)
- def `visualize_anomaly_prediction` L292-L408
  - doc: Anomaly Detection 예측 결과 시각화
- def `visualize_predictions_from_runner` L411-L551
  - doc: 학습된 모델로 각 카테고리별 예측 및 시각화 (선택된 샘플만 inference)
- def `visualize_single_prediction` L554-L656
  - doc: 단일 카테고리의 특정 샘플 시각화 (간편 함수)