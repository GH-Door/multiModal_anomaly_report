# AD-MLLM 인수인계 개요 (공통)

## 1. 최근 코드 변경사항 (핵심)
1) AD 추론 엔트리 통합
- 기존 분리 스크립트 대신 `scripts/run_ad_inference.py` 중심으로 정리
- `ckpt`/`onnx` 백엔드 모두 이 스크립트에서 처리

2) AD JSON 파싱/어댑터 모듈화
- `src/ad/io.py`: AD 결과 JSON 로드/정규화
- `src/ad/adapter.py`: AD 결과를 LLM 입력 포맷으로 변환
- 목적: AD 출력 구조가 바뀌어도 실험/서비스 코드 영향 최소화

3) run_experiment의 AD 결합 방식 개선
- `scripts/run_experiment.py`가 AD JSON 재사용(`--ad-output`) 또는 AD 추론 자동 실행을 지원
- AD 결과를 image_path 기준으로 매칭 후 `to_llm_ad_info()`로 LLM에 전달

4) PatchCore-only 학습 강제
- `scripts/train_anomalib.py`는 이제 `anomaly.model == patchcore`만 허용

5) 데이터로더 확장자 버그 수정
- `src/datasets/dataloader.py`의 GoodsAD 학습 이미지 로더가 `jpg` 고정이던 문제 수정
- 현재 `jpg/jpeg/png/bmp` 지원

## 2. 전체 구조
- AD 학습: `scripts/train_anomalib.py` + `src/datasets/dataloader.py`
- AD 추론(JSON 생성): `scripts/run_ad_inference.py`
- AD JSON 로드/변환: `src/ad/io.py`, `src/ad/adapter.py`
- LLM 평가(MCQ): `scripts/run_experiment.py`, `src/mllm/*`, `src/eval/metrics.py`
- 리포트 생성 파이프라인(서비스 참고): `src/mllm/base.py`의 `generate_report()` 계열

## 3. 데이터 흐름 (실험 기준)
1) MMAD 이미지 메타 로드
2) 샘플링(`sample_per_folder`, `max_images`)
3) AD 추론(JSON 생성 또는 재사용)
4) AD JSON -> `to_llm_ad_info()`
5) LLM 질문 응답 생성
6) MCQ 정확도 계산/저장

## 4. 공통 실행 예시
```bash
# End-to-end (AD + LLM)
python scripts/run_experiment.py --config configs/experiment.yaml

# AD 결과 재사용
python scripts/run_experiment.py --config configs/experiment.yaml --ad-output outputs/eval/patchcore_predictions.json

# AD만 실행
python scripts/run_ad_inference.py \
  --backend ckpt \
  --checkpoint-dir /path/to/checkpoints \
  --data-root /path/to/MMAD \
  --mmad-json /path/to/mmad.json \
  --output outputs/eval/patchcore_predictions.json \
  --output-format report
```

## 5. 자주 발생하는 이슈
1) `Available models: 0`
- 원인: 체크포인트 버전(`ad.version`) 불일치
- 조치: `ad.version: null` 또는 실제 버전으로 지정

2) `num_samples=0` (학습)
- 원인: train split 비었거나 확장자/경로 불일치
- 조치: `GoodsAD/<cat>/train/good` 파일 개수 확인

3) InternVL 로딩 오류(`all_tied_weights_keys`)
- 원인: transformers 호환 이슈
- 조치: 코랩에서 `transformers==4.52.4` 고정 후 런타임 재시작
