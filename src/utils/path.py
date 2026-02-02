"""프로젝트 경로 유틸리티

❗ 이 파일은 *환경 자동 판별(Colab/로컬 체크)*이나 *하드코딩 경로*를 사용하지 않습니다.

프로젝트 루트는 다음 우선순위로 결정합니다.

1) 환경변수 ``MMAD_PROJECT_ROOT`` 가 있으면 그 값을 사용
2) 없으면, 현재 파일 위치(src/utils/path.py) 기준으로 상위 2단계(= repo root)를 사용

※ 어떤 검증도 하지 않습니다(요청사항: 환경 체크 제거).
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """프로젝트 루트 경로 반환.

    Notes:
        - 하드코딩/환경 자동판별을 하지 않습니다.
        - 필요 시 환경변수 ``MMAD_PROJECT_ROOT`` 로 강제 지정하세요.
    """
    env_root = os.environ.get("MMAD_PROJECT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # src/utils/path.py -> src/utils -> src -> project_root
    return Path(__file__).resolve().parents[2]


def get_logs_dir() -> Path:
    """로그 디렉토리 경로 반환 (없으면 생성)."""
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_checkpoints_dir() -> Path:
    """체크포인트 디렉토리 경로 반환 (없으면 생성)."""
    ckpt_dir = get_project_root() / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def get_outputs_dir() -> Path:
    """출력 디렉토리 경로 반환 (없으면 생성)."""
    outputs_dir = get_project_root() / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir
