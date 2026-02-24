# ── API 서버 Dockerfile ────────────────────────────────────────────────────────
# 베이스 이미지: PyTorch 2.5.1 + CUDA 12.1 + Python 3.11
#   - Gemma3 INT4 (로컬 모델) 사용 시 GPU 필요
#   - Gemini API 사용 시 GPU 없어도 동작
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# OpenCV, git 등 시스템 의존성
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 git curl \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN pip install --no-cache-dir uv

# torchao: Gemma3 INT4 모델(pytorch/gemma-3-*-INT4) 로드에 필요
RUN uv pip install --system --no-cache torchao

WORKDIR /app

# 의존성 파일 먼저 복사 → 소스 변경 시 레이어 캐시 재사용
COPY pyproject.toml uv.lock ./
COPY src/ src/

# pyproject.toml 기반 설치
# --no-build-isolation: 이미 설치된 torch(베이스 이미지)를 재사용
RUN uv pip install --system --no-cache -e . --no-build-isolation

# 소스 전체 복사
COPY . .

EXPOSE 8000
CMD ["uvicorn", "apps.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
