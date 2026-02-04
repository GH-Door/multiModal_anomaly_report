import yaml
import json
import pandas as pd
import logging
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_env():
    load_dotenv()

def load_config(config_path: str) -> dict[str, Any]:
    """YAML 설정 파일 로드"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        logger.info(f"Loaded config from {config_path}")
        return config

def load_json(json_path: str) -> dict[str, Any]:
    """JSON 파일 로드"""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_csv(csv_path: str) -> pd.DataFrame:
    """CSV 파일 로드"""
    return pd.read_csv(csv_path)
