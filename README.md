<div align="center">

# 🎬 Demo

> 🎥 **Web Demo 영상 제작 중입니다.**

<br>
<br>

# Smart Factory Anomaly Reporting System

**멀티모달 이상 탐지 리포트 자동 생성 시스템**

<br>

# 🏅 Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python_≥3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Anomalib](https://img.shields.io/badge/Anomalib-FF6B35?style=for-the-badge&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

</div>

<br>

## 👥 Team

| ![이호욱](https://github.com/adhoc0909.png) | ![문국현](https://github.com/GH-Door.png) | ![woojeong01](https://github.com/woojeong01.png) | ![0Devilkitty0](https://github.com/0Devilkitty0.png) | ![yeony3436-aa](https://github.com/yeony3436-aa.png) |
| :--: | :--: | :--: | :--: | :--: |
| [이호욱](https://github.com/adhoc0909) | [문국현](https://github.com/GH-Door) | [woojeong01](https://github.com/woojeong01) | [0Devilkitty0](https://github.com/0Devilkitty0) | [yeony3436-aa](https://github.com/yeony3436-aa) |
| 팀장 | 팀원 | 팀원 | 팀원 | 팀원 |

<br>

## Project Overview

| Item | Content |
|:-----|:--------|
| **📅 Period** | 2026.01 ~ 2026.02 |
| **👥 Type** | Team Project |
| **🎯 Goal** | Anomaly Detection + LLM 기반 Defect Report 자동 생성 End-to-end 시스템 |
| **🤖 AD Model** | PatchCore / EfficientAD / WinCLIP (Anomalib) |
| **💬 LLM** | GPT-4o · Claude Sonnet · Gemini 2.5 · InternVL · Gemma3 · Qwen |
| **📊 Benchmark** | [MMAD](https://arxiv.org/abs/2410.09453) — GoodsAD (6 classes) · MVTec-LOCO (4 classes, `splicing_connectors` 제외) |

<br>

## Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Benchmark Results](#-benchmark-results)
- [System Architecture](#️-system-architecture)
- [RAG Pipeline](#-rag-pipeline)
- [Supported Models](#-supported-models)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)
- [License](#-license)

<br>

---

## 🎯 Introduction

### 배경 — 제조업 생산성 위기

한국 제조업은 GDP의 **28.4%** 를 차지하며 (독일·일본 대비 높은 비중), 경제 전반의 핵심 축입니다.
그러나 노동생산성 증가율이 **6.3%p 하락**하며 주요국 중 가장 큰 둔화를 보이고 있고,
품질 검사 자동화는 생산성 회복의 핵심 과제로 부상하고 있습니다.

### 기술적 한계 — 기존 검사 시스템의 3가지 문제

| # | 문제 | 설명 |
|:-:|:-----|:-----|
| 1 | **Heatmap/Mask 출력에 그침** | AD 모델은 이상 위치만 표시할 뿐, Defect 원인·조치 설명을 제공하지 못함 |
| 2 | **라인 변경 시 재학습 필수** | 새 제품·카테고리마다 모델을 재학습해야 해 유지보수 비용이 높음 |
| 3 | **미세 Defect 판단 정확도 부족** | 유사 Defect 간 판별 정확도가 낮아 현장 신뢰도 저하 |

### Solution — Defect 탐지부터 조치까지 원스톱 자동화

**Smart Factory Anomaly Reporting System**은 위 세 가지 한계를 동시에 극복하는 End-to-end AI 검사 Pipeline입니다.

- **탐지를 넘어선 Report**: Defect 원인·위치·조치까지 포함한 한국어 Report 자동 생성
- **재학습 없는 즉시 대응**: Knowledge RAG + Visual RAG로 새로운 제품 카테고리에 즉시 적용
- **의사결정 시간 단축**: 공정 관리자가 Heatmap과 LLM Report를 단일 Dashboard에서 확인

<br>

---

## ✨ Key Features

- 🏆 **MMAD Benchmark 기반 평가**: 논문 기준 MCQ Protocol로 정량 평가, **RAG 적용 시 논문 SOTA (GPT-4o 74.9%) 상회**
- 🔎 **Dual RAG**: Knowledge RAG (Domain Knowledge → Chroma) + Visual RAG (DINOv2 → few-shot)
- 🧠 **Policy-based AD Decision**: `ad_policy.json` → `normal / review_needed / anomaly` 3단계 자동 결정
- ⚡ **Async Pipeline**: ThreadPoolExecutor + PostgreSQL 상태 추적 + 120초 Watchdog
- 🔌 **Multi-LLM Support**: API (GPT-4o / Claude / Gemini) + Local (InternVL / Gemma3 / Qwen / LLaVA)
- 🐳 **One-click Docker Deploy**: PostgreSQL + FastAPI + React/nginx 통합 Stack

<br>

---

## 📊 Benchmark Results

[MMAD 논문](https://arxiv.org/abs/2410.09453)의 MCQ Evaluation Protocol 기준으로 평가한 결과입니다.
**Knowledge RAG + Visual RAG 적용 시, 논문에서 보고된 GPT-4o 최고 성능(74.9%)을 상회합니다.**

> 평가 조건: GoodsAD (6 classes) + MVTec-LOCO (4 classes, `splicing_connectors` 제외), 총 99개 이미지, 1-shot

### Overall MCQ Accuracy

| Model | AD Model | RAG | Accuracy | vs. Paper GPT-4o |
|:------|:---------|:---:|:--------:|:----------------:|
| GPT-4o *(Paper SOTA)* | — | ✗ | 74.9% | baseline |
| Gemini 2.5 Flash Lite | PatchCore | ✗ | 66.89% | -8.01%p |
| Gemma3-27B INT4 | PatchCore | ✗ | 69.56% | -5.34%p |
| Gemini 2.5 Flash Lite | PatchCore | ✅ | 74.44% | -0.46%p |
| **Gemma3-27B INT4** | **PatchCore** | ✅ | **75.11%** | **+0.21%p ↑** |

### Per-Task Accuracy — Gemma3-27B INT4 + PatchCore + RAG

| Dataset | Anomaly Det. | Object Cls. | Object Anal. | Defect Cls. | Defect Loc. | Defect Desc. | Defect Anal. | **Avg** |
|:--------|:------------:|:-----------:|:------------:|:-----------:|:-----------:|:------------:|:------------:|:-------:|
| GoodsAD | 62.5% | 80.0% | 75.6% | 44.1% | 50.0% | 71.4% | 90.7% | **67.8%** |
| MVTec-LOCO | 77.1% | 50.0% | 80.0% | 37.5% | 65.2% | 65.2% | 81.0% | **65.1%** |
| **Average** | **69.8%** | **65.0%** | **77.8%** | **40.8%** | **57.6%** | **68.3%** | **85.8%** | **66.4%** |

> **Key Finding**: RAG 적용으로 Gemma3-27B **+5.55%p**, Gemini 2.5 Flash Lite **+7.55%p** 성능 향상.
> Local 오픈소스 모델(Gemma3-27B INT4)이 API 기반 GPT-4o를 상회하는 성능 달성.

<br>

---

## 🏗️ System Architecture

```
  Input Image
      │
      ▼
┌─────────────┐   score/heatmap  ┌──────────────────┐
│  AdService  │ ───────────────▶ │  Policy Engine   │  normal / review_needed / anomaly
│ (PatchCore) │                  │ (ad_policy.json) │
└─────────────┘                  └──────────────────┘
      │                                   │
      ▼                                   ▼
┌─────────────┐                 ┌───────────────────┐      ┌───────────────────────┐
│   Defect    │                 │    LLM Service    │◀─────│       Dual RAG        │
│ Structuring │──── context ───▶│ (Gemma3 / GPT-4o │      │  ┌─────────────────┐  │
│  (Heatmap   │                 │  / Gemini ...)    │      │  │  Knowledge RAG  │  │
│   → JSON)   │                 └───────────────────┘      │  │ (Chroma+Domain) │  │
└─────────────┘                          │                 │  ├─────────────────┤  │
                                         ▼                 │  │   Visual RAG    │  │
                                  ┌─────────────┐          │  │ (DINOv2+few-shot│  │
                                  │   Report    │          │  └─────────────────┘  │
                                  │   (JSON)    │          └───────────────────────┘
                                  └─────────────┘
                                         │
                                         ▼
                                  ┌─────────────┐
                                  │ PostgreSQL  │
                                  └─────────────┘
```

**Production API Pipeline** (비동기):
`POST /inspect` → `AdService.predict_batch()` → PostgreSQL 초기 저장 → ThreadPoolExecutor (RAG + LLM) → PostgreSQL 최종 업데이트

<br>

---

## 🔎 RAG Pipeline

### Knowledge RAG

`domain_knowledge.json` (`{dataset → category → defect_type → description}`) → Chroma Vector DB → Metadata Filter + Semantic Search → Prompt Injection

- Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Vector DB: Chroma (local persist, `vectorstore/domain_knowledge/`)

### Visual RAG

DINOv2 (`dinov2_vits14`) Embedding → 카테고리별 `.pkl` Index → top-k 유사 정상 이미지 few-shot 제공

<br>

---

## 🤖 Supported Models

| Type | Model Key | Description |
|:-----|:----------|:------------|
| **API** | `gpt-4o`, `gpt-4o-mini` | OpenAI |
| **API** | `claude` | Anthropic Claude Sonnet 4 |
| **API** | `gemini-2.5-flash`, `gemini-2.5-pro` | Google Gemini |
| **Local** | `internvl`, `internvl3.5-2b` | InternVL3.5 (1B~8B) |
| **Local** | `gemma3`, `gemma3-12b-int4` | Gemma3 (4B/12B/27B, INT4/INT8) |
| **Local** | `qwen`, `qwen-7b` | Qwen2.5-VL / Qwen3-VL |
| **Local** | `llava` | LLaVA v1.5/v1.6 |

<br>

---

## 🛠️ Installation

### Requirements

- Python ≥ 3.10
- CUDA 12.1+ (GPU 추론 시) / CPU 가능
- Docker & Docker Compose (배포 시)
- PostgreSQL (API Server 실행 시)

### 1. Clone Repository

```bash
git clone https://github.com/<org>/smart-factory-anomaly-report.git
cd smart-factory-anomaly-report
```

### 2. Install Dependencies

```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Lockfile 기반 재현 가능 설치
uv sync --frozen
```

### 3. Set Environment Variables

```bash
cp .env.example .env
# .env 편집: DATABASE_URL, API Key 등
```

| Variable | Default | Description |
|:---------|:--------|:------------|
| `DATABASE_URL` | `postgresql://localhost/inspection` | PostgreSQL 연결 |
| `LLM_MODEL` | `internvl` | 기본 LLM Model |
| `AD_CHECKPOINT_DIR` | `checkpoints/` | PatchCore Checkpoint 경로 |
| `RAG_ENABLED` | `true` | Visual RAG 활성화 |
| `DOMAIN_RAG_ENABLED` | `true` | Knowledge RAG 활성화 |

### 4. Run with Docker (권장)

```bash
docker compose up -d
```

<br>

---

## 🚀 Usage

### AD Model Training

```bash
python scripts/train_anomalib.py   # configs/anomaly.yaml 기준
```

### MMAD Benchmark Evaluation

```bash
# LLM + AD + RAG 조합
python scripts/run_experiment.py --llm gpt-4o --ad-model patchcore --rag

# LLM Only (AD 없음)
python scripts/run_experiment.py --llm gemini-2.5-flash --ad-model null

# 지원 Model 목록 확인
python scripts/run_experiment.py --list-models
```

### Run API Server

```bash
# FastAPI (Port 8000)
uvicorn apps.api.main:app --reload --port 8000

# React Frontend (Port 5173)
cd apps/frontend && npm ci && npm run dev -- --host 0.0.0.0 --port 5173
```

<br>

---

## 📁 Project Structure

```
smart-factory-anomaly-report/
├── apps/
│   ├── api/              # FastAPI Server (Production Pipeline)
│   ├── dashboard/        # Streamlit UI
│   └── frontend/         # React Frontend
├── configs/
│   ├── experiment.yaml   # MMAD Benchmark 평가 설정
│   ├── anomaly.yaml      # Anomalib Training 설정
│   └── ad_policy.json    # AD Decision Threshold (3-tier)
├── docs/                 # 배포 · 실험 · Pipeline 상세 문서
├── scripts/              # Training / Inference / Evaluation CLI
├── src/
│   ├── mllm/             # MLLM Client + Factory
│   ├── rag/              # Knowledge RAG (Chroma) + Visual RAG (DINOv2)
│   ├── service/          # AdService · LlmService · Pipeline
│   ├── storage/          # PostgreSQL / SQLite
│   ├── structure/        # Heatmap → Structured Defect
│   └── eval/             # AUROC · PRO · Dice · IoU Metrics
├── docker-compose.yml
└── pyproject.toml
```

<br>

---

## 📄 Documentation

| Document | Description |
|:---------|:------------|
| [`docs/deploy-and-handoff.md`](docs/deploy-and-handoff.md) | 서버 배포 및 인수인계 가이드 |
| [`docs/experiment-runner.md`](docs/experiment-runner.md) | Benchmark 실험 설정 상세 |
| [`docs/report-pipeline-guide.md`](docs/report-pipeline-guide.md) | Report 생성 Pipeline 상세 |
| [`docs/incoming-auto-ingest.md`](docs/incoming-auto-ingest.md) | Filesystem Auto-Ingest 설정 |

**Reference Paper**: [MMAD: The First-Ever Comprehensive Benchmark for Multimodal LLMs in the Industrial Anomaly Detection Domain](https://arxiv.org/abs/2410.09453)

<br>

---

## 📝 License

This project is licensed under the MIT License.

---

<div align="center">
Made with ❤️ by Likelion AI School Team
</div>
