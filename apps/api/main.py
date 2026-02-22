from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import src.storage.pg as pg
from src.mllm.factory import get_llm_client, list_llm_models
from src.service.ad_service import AdService
from src.service.llm_service import LlmService
from src.service.visual_rag_service import RagService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", os.getenv("PG_DSN", "postgresql://son:1234@localhost/inspection"))
MODEL_NAME = os.getenv("LLM_MODEL", "internvl")
CHECKPOINT_DIR = os.getenv("AD_CHECKPOINT_DIR", "/home/ubuntu/apps/checkpoints")
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "/home/ubuntu/apps/rag")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/home/ubuntu/ad_results"))
DATA_DIR = Path(os.getenv("DATA_DIR", "/home/ubuntu/dataset"))


class LlmModelUpdateRequest(BaseModel):
    model: str


def _set_llm_model(app: FastAPI, model_name: str) -> None:
    selected = model_name.strip()
    if not selected:
        raise ValueError("Model name must not be empty")
    app.state.llm_client = get_llm_client(selected)
    app.state.llm_service = LlmService(client=app.state.llm_client)
    app.state.llm_model = selected


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_conn = pg.connect(DATABASE_URL)
    app.state.ad_service = AdService(
        checkpoint_dir=CHECKPOINT_DIR,
        output_root=str(OUTPUT_DIR),
    )
    app.state.rag_service = RagService(index_dir=RAG_INDEX_DIR)
    _set_llm_model(app, MODEL_NAME)
    try:
        yield
    finally:
        app.state.db_conn.close()


app = FastAPI(title="Industrial AI Inspection System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR), check_dir=False), name="outputs")
app.mount("/data/datasets", StaticFiles(directory=str(DATA_DIR), check_dir=False), name="datasets")


@app.get("/settings/llm-model")
async def get_llm_model_settings():
    return {
        "active_model": str(getattr(app.state, "llm_model", MODEL_NAME)),
        "available_models": list_llm_models(),
    }


@app.put("/settings/llm-model")
async def update_llm_model_settings(payload: LlmModelUpdateRequest):
    model_name = payload.model.strip()
    try:
        _set_llm_model(app, model_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to switch LLM model")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "active_model": str(app.state.llm_model),
        "available_models": list_llm_models(),
    }


@app.post("/inspect")
async def run_inspection(
    file: UploadFile = File(...),
    category: str = Form(...),
    dataset: str = Form("default"),
    line: str = Form("line_1"),
):
    """Run AD -> RAG -> LLM and persist report fields used by current frontend."""
    conn = app.state.db_conn
    try:
        policy = pg.get_category_policy(conn, category)
        t_low = float(policy.get("t_low", 0.5))
        t_high = float(policy.get("t_high", 0.8))

        ad_start_time = datetime.now()
        ad_results = app.state.ad_service.predict_batch(
            [file],
            category=category,
            dataset=dataset,
            threshold=t_low,
        )
        ad_result = ad_results[0]
        score = float(ad_result["ad_score"])

        if score < t_low:
            ad_decision = "normal"
        elif score > t_high:
            ad_decision = "anomaly"
        else:
            ad_decision = "review_needed"

        initial_data = {
            "dataset": dataset,
            "category": category,
            "line": line,
            "ad_score": score,
            "ad_decision": ad_decision,
            "is_anomaly_ad": score > t_high,
            "has_defect": ad_result.get("has_defect", False),
            "region": ad_result.get("region"),
            "area_ratio": ad_result.get("area_ratio"),
            "bbox": ad_result.get("bbox"),
            "center": ad_result.get("center"),
            "confidence": ad_result.get("confidence"),
            "image_path": ad_result["original_path"],
            "heatmap_path": ad_result["heatmap_path"],
            "mask_path": ad_result["mask_path"],
            "overlay_path": ad_result["overlay_path"],
            "created_at": ad_start_time,
            "ad_inference_duration": (datetime.now() - ad_start_time).total_seconds(),
            "applied_policy": policy,
        }
        report_id = pg.insert_report(conn, initial_data)

        rag_results = app.state.rag_service.search_closest_normal(ad_result["original_path"], category)
        ref_path = rag_results[0]["path"] if rag_results else None
        pg.update_report(conn, report_id, {"similar_image_path": ref_path})

        llm_response = app.state.llm_service.generate_dynamic_report(
            ad_result["original_path"],
            ref_path,
            category,
            {**ad_result, "ad_decision": ad_decision},
            policy,
        )
        pg.update_report(conn, report_id, llm_response)

        return {"status": "success", "report_id": report_id, "ad_decision": ad_decision}
    except Exception as exc:
        logger.exception("Inspection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reports")
async def get_reports(
    category: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(5000, ge=1, le=5000),
):
    """Fetch reports in the same shape expected by current frontend."""
    offset = (page - 1) * page_size
    try:
        reports = pg.get_filtered_reports(
            app.state.db_conn,
            category=category,
            decision=decision,
            limit=page_size,
            offset=offset,
        )
        return {
            "page": page,
            "page_size": page_size,
            "total_count": len(reports),
            "items": reports,
        }
    except Exception as exc:
        logger.exception("Report list fetch failed")
        raise HTTPException(status_code=500, detail="데이터 조회 중 오류가 발생했습니다.") from exc
